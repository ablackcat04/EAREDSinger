"""Build a song-level dataset for the accompaniment tension refiner.

The frozen variance checkpoint is run in inference mode on the existing
sentence-level binary dataset. Predictions and ground truth curves are then
placed back on the original full-song timeline using the sentence TextGrid.
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import torch


ROOT_DIR = Path(__file__).parent.parent.resolve()
os.environ["PYTHONPATH"] = str(ROOT_DIR)
sys.path.insert(0, str(ROOT_DIR))

import utils
from modules.toplevel import DiffSingerVariance
from training.variance_task import VarianceDataset
from utils.accompaniment_timeline import (
    load_song_segment_timings,
    song_id_from_segment_name,
    strip_dataset_prefix,
)
from utils.binarizer_utils import get_mel_torch
from utils.hparams import hparams, set_hparams
from utils.indexed_datasets import IndexedDatasetBuilder
from utils.phoneme_utils import load_phoneme_dictionary


def _script_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args, _ = parser.parse_known_args()
    return args


def _move_to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _downsample_mel(mel: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 0:
        raise ValueError("refiner_downsample_factor must be positive.")
    if mel.ndim != 2 or mel.shape[0] == 0:
        raise ValueError(f"Expected a non-empty mel [T, M], received {mel.shape}.")
    padded_length = ((mel.shape[0] + factor - 1) // factor) * factor
    if padded_length != mel.shape[0]:
        mel = np.pad(mel, ((0, padded_length - mel.shape[0]), (0, 0)), mode="edge")
    return mel.reshape(-1, factor, mel.shape[1]).mean(axis=1).astype(np.float32)


def _predict_base_tension(model: DiffSingerVariance, sample: dict) -> torch.Tensor:
    _, _, variances = model(
        sample["tokens"],
        languages=sample.get("languages"),
        midi=sample.get("midi"),
        ph2word=sample.get("ph2word"),
        ph_dur=sample["ph_dur"],
        mel2ph=sample.get("mel2ph"),
        note_midi=sample.get("note_midi"),
        note_rest=sample.get("note_rest"),
        note_dur=sample.get("note_dur"),
        note_glide=sample.get("note_glide"),
        mel2note=sample.get("mel2note"),
        base_pitch=sample.get("base_pitch"),
        # Deliberately omit ground-truth pitch and variance curves. The refiner
        # must learn from the same base predictions it will receive at inference.
        pitch=None,
        spk_id=sample.get("spk_ids"),
        infer=True,
    )
    if variances is None or "tension" not in variances:
        raise RuntimeError("The base variance model did not return a tension prediction.")
    return variances["tension"]


def _copy_training_payload(source_dir: Path, output_dir: Path) -> None:
    for filename in ("spk_map.json", "lang_map.json"):
        source = source_dir / filename
        if source.exists():
            shutil.copy2(source, output_dir / filename)
    for source in source_dir.glob("dictionary-*.txt"):
        shutil.copy2(source, output_dir / source.name)


def _build_prefix(
    prefix: str,
    dataset: VarianceDataset,
    model: DiffSingerVariance,
    device: torch.device,
    output_dir: Path,
    accompaniment_dir: Path,
    textgrid_dir: Path,
    downsample_factor: int,
    sentence_tier: str,
    overwrite: bool,
) -> None:
    output_data = output_dir / f"{prefix}.data"
    output_meta = output_dir / f"{prefix}.meta"
    existing = [path for path in (output_data, output_meta) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Refusing to replace {existing[0]}; pass --overwrite to rebuild the dataset."
        )
    for path in existing:
        path.unlink()

    records_by_song: dict[str, list[dict]] = defaultdict(list)
    print(f"| Generating frozen base tension for {len(dataset)} {prefix} segments.")
    for index in range(len(dataset)):
        raw_item = dataset[index]
        segment_name = strip_dataset_prefix(raw_item["name"])
        song_id = song_id_from_segment_name(segment_name)
        batch = _move_to_device(dataset.collater([raw_item]), device)
        base_prediction = _predict_base_tension(model, batch)[0]
        target = raw_item["tension"]
        length = int(target.shape[0])
        records_by_song[song_id].append(
            {
                "name": segment_name,
                "base": base_prediction[:length].detach().cpu().float().numpy(),
                "target": target[:length].cpu().float().numpy(),
                "curve_mask": (raw_item["mel2ph"][:length] > 0).cpu().numpy(),
                "tension_mask": (
                    (raw_item["mel2ph"][:length] > 0) & ~raw_item["uv"][:length].bool()
                ).cpu().numpy(),
            }
        )

    timestep = hparams["hop_size"] / hparams["audio_sample_rate"]
    metadata = {"names": [], "lengths": [], "mel_accomp_coarse": []}
    builder = IndexedDatasetBuilder(output_dir, prefix=prefix)
    try:
        for song_id in sorted(records_by_song):
            wav_path = accompaniment_dir / f"{song_id}.wav"
            textgrid_path = textgrid_dir / f"{song_id}.TextGrid"
            if not wav_path.is_file():
                raise FileNotFoundError(f"Full normalized accompaniment not found: {wav_path}")
            if not textgrid_path.is_file():
                raise FileNotFoundError(f"Full-song TextGrid not found: {textgrid_path}")

            print(f"| Extracting full-song accompaniment mel: {wav_path.name}")
            waveform, _ = librosa.load(
                wav_path, sr=hparams["audio_sample_rate"], mono=True
            )
            full_mel = get_mel_torch(
                waveform,
                samplerate=hparams["audio_sample_rate"],
                num_mel_bins=hparams["accompaniment_num_mel_bins"],
                hop_size=hparams["accompaniment_mel_hop_size"],
                win_size=hparams["accompaniment_mel_win_size"],
                fft_size=hparams["accompaniment_mel_fft_size"],
                fmin=hparams["accompaniment_mel_fmin"],
                fmax=hparams["accompaniment_mel_fmax"],
            )
            coarse_mel = _downsample_mel(full_mel, downsample_factor)
            full_length = int(full_mel.shape[0])
            base_full = np.zeros(full_length, dtype=np.float32)
            target_full = np.zeros(full_length, dtype=np.float32)
            curve_mask_full = np.zeros(full_length, dtype=np.bool_)
            tension_mask_full = np.zeros(full_length, dtype=np.bool_)
            timings = load_song_segment_timings(
                textgrid_path, sentence_tier=sentence_tier
            )

            for record in records_by_song[song_id]:
                if record["name"] not in timings:
                    raise KeyError(
                        f"No full-song timing for segment {record['name']} in {textgrid_path}."
                    )
                timing = timings[record["name"]]
                start = round(timing.start / timestep)
                segment_length = len(record["target"])
                end = start + segment_length
                if start < 0 or end > full_length:
                    raise ValueError(
                        f"Segment {record['name']} maps to frames {start}:{end}, "
                        f"outside the accompaniment length {full_length}."
                    )
                if curve_mask_full[start:end].any():
                    raise ValueError(f"Overlapping vocal segments around {record['name']}.")
                base_full[start:end] = record["base"]
                target_full[start:end] = record["target"]
                curve_mask_full[start:end] = record["curve_mask"]
                tension_mask_full[start:end] = record["tension_mask"]

            if not tension_mask_full.any():
                raise ValueError(f"Song {song_id} contains no valid tension target frames.")
            builder.add_item(
                {
                    "song_id": int(song_id) if song_id.isdigit() else song_id.encode("utf-8"),
                    "mel_accomp_coarse": coarse_mel,
                    "base_tension": base_full,
                    "target_tension": target_full,
                    "curve_mask": curve_mask_full,
                    "tension_mask": tension_mask_full,
                }
            )
            metadata["names"].append(song_id)
            metadata["lengths"].append(full_length)
            metadata["mel_accomp_coarse"].append(len(coarse_mel))
    finally:
        builder.finalize()

    with output_meta.open("wb") as file:
        pickle.dump(metadata, file)
    print(f"| Wrote {len(metadata['names'])} full-song {prefix} examples to {output_dir}.")


def main() -> None:
    set_hparams()
    args = _script_args()
    if hparams.get("use_accompaniment", False):
        raise ValueError(
            "The base model for refiner preparation must have use_accompaniment: false."
        )
    checkpoint = hparams.get("base_variance_ckpt")
    if not checkpoint:
        raise ValueError("Set base_variance_ckpt in the refiner config before preparation.")

    random.seed(hparams.get("refiner_prediction_seed", 1234))
    np.random.seed(hparams.get("refiner_prediction_seed", 1234))
    torch.manual_seed(hparams.get("refiner_prediction_seed", 1234))
    device = torch.device(args.device)
    source_binary_dir = Path(hparams["base_binary_data_dir"])
    output_dir = Path(hparams["binary_data_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    original_binary_dir = hparams["binary_data_dir"]
    hparams["binary_data_dir"] = str(source_binary_dir)
    try:
        datasets = {
            "train": VarianceDataset("train"),
            "valid": VarianceDataset("valid"),
        }
    finally:
        hparams["binary_data_dir"] = original_binary_dir

    dictionary = load_phoneme_dictionary()
    model = DiffSingerVariance(vocab_size=len(dictionary)).to(device).eval()
    utils.load_ckpt(model, checkpoint, device=device, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    accompaniment_dir = Path(hparams["refiner_accompaniment_dir"])
    textgrid_dir = Path(hparams["refiner_textgrid_dir"])
    with torch.inference_mode():
        for prefix, dataset in datasets.items():
            _build_prefix(
                prefix,
                dataset,
                model,
                device,
                output_dir,
                accompaniment_dir,
                textgrid_dir,
                hparams["refiner_downsample_factor"],
                hparams["refiner_sentence_tier"],
                args.overwrite,
            )
    _copy_training_payload(source_binary_dir, output_dir)


if __name__ == "__main__":
    main()
