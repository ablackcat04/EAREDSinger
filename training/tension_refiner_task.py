from __future__ import annotations

import pickle
from pathlib import Path

import torch

import utils
from basics.base_dataset import BaseDataset
from basics.base_task import BaseTask
from modules.accompaniment_refiner import AccompanimentTensionRefiner
from modules.metrics import RawCurveR2Score
from utils.hparams import hparams
from utils.plot import curve_to_figure


class TensionRefinerDataset(BaseDataset):
    def collater(self, samples):
        batch = super().collater(samples)
        if batch["size"] == 0:
            return batch
        if "mel_accomp" not in samples[0]:
            raise RuntimeError(
                "This refiner dataset contains legacy downsampled mels. Re-run "
                "scripts/prepare_tension_refiner.py --overwrite to store full-resolution mels."
            )
        mel_lengths = torch.LongTensor([len(sample["mel_accomp"]) for sample in samples])
        for sample, mel_length in zip(samples, mel_lengths.tolist()):
            if len(sample["base_tension"]) != mel_length:
                raise ValueError(
                    "Full-resolution accompaniment mel and tension lengths must match."
                )
        batch.update(
            {
                "mel_accomp": utils.collate_nd(
                    [sample["mel_accomp"] for sample in samples], 0.0
                ),
                "mel_lengths": mel_lengths,
                "base_tension": utils.collate_nd(
                    [sample["base_tension"] for sample in samples], 0.0
                ),
                "target_tension": utils.collate_nd(
                    [sample["target_tension"] for sample in samples], 0.0
                ),
                "curve_mask": utils.collate_nd(
                    [sample["curve_mask"] for sample in samples], False
                ),
                "tension_mask": utils.collate_nd(
                    [sample["tension_mask"] for sample in samples], False
                ),
            }
        )
        return batch


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = value[mask]
    if selected.numel() == 0:
        return value.sum() * 0.0
    return selected.mean()


def _load_mel_normalization() -> dict:
    metadata_path = Path(hparams["binary_data_dir"]) / "train.meta"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Refiner metadata not found: {metadata_path}. Prepare the dataset first."
        )
    with metadata_path.open("rb") as file:
        metadata = pickle.load(file)
    required = ("mel_mean", "mel_std", "mel_level_mean", "mel_level_std")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise RuntimeError(
            f"Refiner metadata is missing {missing}. Re-run "
            "scripts/prepare_tension_refiner.py --overwrite with the new full-mel pipeline."
        )
    return {key: metadata[key] for key in required}


class TensionRefinerTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = TensionRefinerDataset
        self.lambda_delta = hparams["lambda_refiner_delta"]
        self.lambda_smooth = hparams["lambda_refiner_smooth"]
        super()._finish_init()

    def _build_model(self):
        normalization = _load_mel_normalization()
        return AccompanimentTensionRefiner(
            mel_bins=hparams["accompaniment_num_mel_bins"],
            hidden_size=hparams["refiner_hidden_size"],
            num_layers=hparams["refiner_num_layers"],
            num_heads=hparams["refiner_num_heads"],
            ffn_size=hparams["refiner_ffn_size"],
            downsample_factor=hparams["refiner_downsample_factor"],
            num_local_layers=hparams["refiner_local_layers"],
            local_kernel_size=hparams["refiner_local_kernel_size"],
            dropout=hparams["refiner_dropout"],
            initial_gate=hparams["refiner_initial_gate"],
            mel_timestep=(
                hparams["accompaniment_mel_hop_size"]
                / hparams["audio_sample_rate"]
            ),
            time_scale_seconds=hparams["refiner_time_scale_seconds"],
            tension_min=hparams["tension_logit_min"],
            tension_max=hparams["tension_logit_max"],
            **normalization,
        )

    def build_losses_and_metrics(self):
        self.register_validation_loss("tension_loss")
        self.register_validation_loss("delta_reg_loss")
        self.register_validation_loss("smoothness_loss")
        self.register_validation_metric("base_tension_r2", RawCurveR2Score())
        self.register_validation_metric("refined_tension_r2", RawCurveR2Score())

    def run_model(self, sample, infer=False):
        refined, delta, coarse_delta = self.model(
            sample["mel_accomp"].float(),
            sample["base_tension"],
            sample["curve_mask"],
            sample["mel_lengths"],
        )
        if infer:
            return refined, delta, coarse_delta

        target = sample["target_tension"]
        tension_mask = sample["tension_mask"].bool()
        curve_mask = sample["curve_mask"].bool()
        tension_loss = _masked_mean((refined - target) ** 2, tension_mask)
        delta_reg_loss = self.lambda_delta * _masked_mean(delta.abs(), curve_mask)
        if coarse_delta.shape[1] > 1:
            coarse_lengths = torch.div(
                sample["mel_lengths"] + self.model.downsample_factor - 1,
                self.model.downsample_factor,
                rounding_mode="floor",
            )
            pair_index = torch.arange(
                coarse_delta.shape[1] - 1, device=coarse_delta.device
            )[None]
            pair_mask = pair_index < (coarse_lengths[:, None] - 1)
            smoothness_loss = self.lambda_smooth * _masked_mean(
                (coarse_delta[:, 1:] - coarse_delta[:, :-1]).abs(), pair_mask
            )
        else:
            smoothness_loss = coarse_delta.sum() * 0.0
        return {
            "tension_loss": tension_loss,
            "delta_reg_loss": delta_reg_loss,
            "smoothness_loss": smoothness_loss,
        }

    def _validation_step(self, sample, batch_idx):
        losses = self.run_model(sample, infer=False)
        refined, delta, _ = self.run_model(sample, infer=True)
        mask = sample["tension_mask"].bool()
        self.valid_metrics["base_tension_r2"].update(
            pred=sample["base_tension"],
            target=sample["target_tension"],
            mask=mask,
        )
        self.valid_metrics["refined_tension_r2"].update(
            pred=refined, target=sample["target_tension"], mask=mask
        )

        for batch_item, data_idx in enumerate(sample["indices"].tolist()):
            if data_idx >= hparams["num_valid_plots"]:
                continue
            length = self.valid_dataset.metadata["lengths"][data_idx]
            figure = curve_to_figure(
                sample["target_tension"][batch_item, :length],
                refined[batch_item, :length],
                sample["base_tension"][batch_item, :length],
                title=self.valid_dataset.metadata["names"][data_idx],
            )
            self.logger.all_rank_experiment.add_figure(
                f"tension_refiner_{data_idx}", figure, self.global_step
            )
            delta_figure = curve_to_figure(
                torch.zeros_like(delta[batch_item, :length]),
                delta[batch_item, :length],
                title=(
                    f"{self.valid_dataset.metadata['names'][data_idx]} "
                    "- accompaniment delta"
                ),
            )
            self.logger.all_rank_experiment.add_figure(
                f"tension_delta_{data_idx}", delta_figure, self.global_step
            )
        return losses, sample["size"]
