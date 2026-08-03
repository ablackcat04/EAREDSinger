from __future__ import annotations

import torch
import torch.nn.functional as F

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
        batch.update(
            {
                "mel_accomp_coarse": utils.collate_nd(
                    [sample["mel_accomp_coarse"] for sample in samples], 0.0
                ),
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


class TensionRefinerTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = TensionRefinerDataset
        self.lambda_delta = hparams["lambda_refiner_delta"]
        self.lambda_smooth = hparams["lambda_refiner_smooth"]
        super()._finish_init()

    def _build_model(self):
        return AccompanimentTensionRefiner(
            mel_bins=hparams["accompaniment_num_mel_bins"],
            hidden_size=hparams["refiner_hidden_size"],
            num_layers=hparams["refiner_num_layers"],
            kernel_size=hparams["refiner_kernel_size"],
            dropout=hparams["refiner_dropout"],
            initial_gate=hparams["refiner_initial_gate"],
            tension_min=hparams["tension_logit_min"],
            tension_max=hparams["tension_logit_max"],
        )

    def build_losses_and_metrics(self):
        self.register_validation_loss("tension_loss")
        self.register_validation_loss("delta_reg_loss")
        self.register_validation_loss("smoothness_loss")
        self.register_validation_metric("base_tension_r2", RawCurveR2Score())
        self.register_validation_metric("refined_tension_r2", RawCurveR2Score())

    def run_model(self, sample, infer=False):
        refined, delta, coarse_delta = self.model(
            sample["mel_accomp_coarse"],
            sample["base_tension"],
            sample["curve_mask"],
        )
        if infer:
            return refined, delta, coarse_delta

        target = sample["target_tension"]
        tension_mask = sample["tension_mask"].bool()
        curve_mask = sample["curve_mask"].bool()
        tension_loss = _masked_mean((refined - target) ** 2, tension_mask)
        delta_reg_loss = self.lambda_delta * _masked_mean(delta.abs(), curve_mask)
        if coarse_delta.shape[1] > 1:
            smoothness_loss = self.lambda_smooth * F.l1_loss(
                coarse_delta[:, 1:], coarse_delta[:, :-1]
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
            pred=sample["base_tension"], target=sample["target_tension"], mask=mask
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
                title=f"{self.valid_dataset.metadata['names'][data_idx]} - accompaniment delta",
            )
            self.logger.all_rank_experiment.add_figure(
                f"tension_delta_{data_idx}", delta_figure, self.global_step
            )
        return losses, sample["size"]
