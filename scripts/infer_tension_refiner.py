"""Refine tension curves in an existing DS using a full accompaniment."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import torch


ROOT_DIR = Path(__file__).parent.parent.resolve()
os.environ["PYTHONPATH"] = str(ROOT_DIR)
sys.path.insert(0, str(ROOT_DIR))

import utils
from modules.accompaniment_refiner import AccompanimentTensionRefiner
from utils.binarizer_utils import get_mel_torch
from utils.hparams import hparams, set_hparams


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a trained accompaniment tension refiner to an existing DS."
    )
    parser.add_argument("--accompaniment", type=Path, required=True)
    parser.add_argument("--ds", type=Path, required=True, help="DS containing base tension curves")
    parser.add_argument("--refiner-ckpt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output refined .ds file")
    parser.add_argument(
        "--diagnostics",
        type=Path,
        help="Optional NPZ containing the full-song base, delta, and refined curves",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args, _ = parser.parse_known_args()
    return args


def _number_sequence(value, *, field: str, segment_index: int) -> np.ndarray:
    if isinstance(value, str):
        values = value.split()
    elif isinstance(value, list):
        values = value
    else:
        raise ValueError(
            f"Segment {segment_index}: {field!r} must be a whitespace string or list."
        )
    try:
        array = np.asarray(values, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Segment {segment_index}: invalid {field!r} values.") from error
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"Segment {segment_index}: {field!r} must contain finite values.")
    return array


def _segment_duration(segment: dict, segment_index: int) -> float:
    field = "ph_dur" if segment.get("ph_dur") not in (None, "") else "note_dur"
    if segment.get(field) in (None, ""):
        raise ValueError(f"Segment {segment_index}: missing ph_dur and note_dur.")
    durations = _number_sequence(segment[field], field=field, segment_index=segment_index)
    if (durations < 0).any() or durations.sum() <= 0:
        raise ValueError(f"Segment {segment_index}: durations must be non-negative and non-empty.")
    return float(durations.sum())


def _resample_curve(
    curve: np.ndarray,
    source_timestep: float,
    target_timestep: float,
    target_length: int,
) -> np.ndarray:
    if target_length <= 0:
        raise ValueError("Curve target length must be positive.")
    if len(curve) == 1:
        return np.full(target_length, curve[0], dtype=np.float32)
    source_times = np.arange(len(curve), dtype=np.float64) * source_timestep
    target_times = np.arange(target_length, dtype=np.float64) * target_timestep
    return np.interp(target_times, source_times, curve).astype(np.float32)


def _load_ds(path: Path) -> tuple[object, list[dict]]:
    try:
        root = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    segments = root if isinstance(root, list) else [root]
    if not segments or not all(isinstance(segment, dict) for segment in segments):
        raise ValueError("The DS root must be an object or a non-empty list of objects.")
    return root, segments


def _assemble_base_tension(
    segments: list[dict], full_length: int, timestep: float
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    base_tension = np.zeros(full_length, dtype=np.float32)
    curve_mask = np.zeros(full_length, dtype=np.bool_)
    layouts: list[dict] = []
    for index, segment in enumerate(segments):
        if segment.get("tension") in (None, ""):
            raise ValueError(
                f"Segment {index}: missing base 'tension'. Run the base variance model first."
            )
        try:
            offset = float(segment.get("offset", 0.0))
            source_timestep = float(segment["tension_timestep"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Segment {index}: valid offset and tension_timestep are required."
            ) from error
        if not math.isfinite(offset) or offset < 0:
            raise ValueError(f"Segment {index}: offset must be non-negative and finite.")
        if not math.isfinite(source_timestep) or source_timestep <= 0:
            raise ValueError(f"Segment {index}: tension_timestep must be positive and finite.")

        original_curve = _number_sequence(
            segment["tension"], field="tension", segment_index=index
        )
        segment_length = round(_segment_duration(segment, index) / timestep)
        start = round(offset / timestep)
        end = start + segment_length
        if end > full_length:
            raise ValueError(
                f"Segment {index}: frames {start}:{end} exceed accompaniment length {full_length}."
            )
        if curve_mask[start:end].any():
            raise ValueError(f"Segment {index}: overlapping DS tension regions are ambiguous.")
        base_tension[start:end] = _resample_curve(
            original_curve, source_timestep, timestep, segment_length
        )
        curve_mask[start:end] = True
        layouts.append(
            {
                "start": start,
                "end": end,
                "source_timestep": source_timestep,
                "source_length": len(original_curve),
            }
        )
    return base_tension, curve_mask, layouts



def _format_curve(curve: np.ndarray) -> str:
    return " ".join(f"{float(value):.6f}" for value in curve)


def main() -> None:
    set_hparams()
    args = _args()
    if args.output.suffix.lower() != ".ds":
        raise ValueError("--output must have a .ds extension.")
    if args.ds.resolve() == args.output.resolve():
        raise ValueError("Input and output DS paths must be different.")
    root, segments = _load_ds(args.ds)

    waveform, _ = librosa.load(
        args.accompaniment, sr=hparams["audio_sample_rate"], mono=True
    )
    mel = get_mel_torch(
        waveform,
        samplerate=hparams["audio_sample_rate"],
        num_mel_bins=hparams["accompaniment_num_mel_bins"],
        hop_size=hparams["accompaniment_mel_hop_size"],
        win_size=hparams["accompaniment_mel_win_size"],
        fft_size=hparams["accompaniment_mel_fft_size"],
        fmin=hparams["accompaniment_mel_fmin"],
        fmax=hparams["accompaniment_mel_fmax"],
    )
    timestep = hparams["hop_size"] / hparams["audio_sample_rate"]
    base_tension, curve_mask, layouts = _assemble_base_tension(
        segments, len(mel), timestep
    )
    device = torch.device(args.device)
    model = AccompanimentTensionRefiner(
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
            hparams["accompaniment_mel_hop_size"] / hparams["audio_sample_rate"]
        ),
        time_scale_seconds=hparams["refiner_time_scale_seconds"],
        tension_min=hparams["tension_logit_min"],
        tension_max=hparams["tension_logit_max"],
    ).to(device).eval()
    utils.load_ckpt(model, args.refiner_ckpt, device=device, strict=True)
    with torch.inference_mode():
        refined, delta, _ = model(
            torch.from_numpy(mel)[None].to(device),
            torch.from_numpy(base_tension)[None].to(device),
            torch.from_numpy(curve_mask)[None].to(device),
            torch.tensor([len(mel)], device=device),
        )
    refined_full = refined[0].cpu().numpy()
    delta_full = delta[0].cpu().numpy()

    output_root = copy.deepcopy(root)
    output_segments = output_root if isinstance(output_root, list) else [output_root]
    for segment, layout in zip(output_segments, layouts):
        refined_segment = refined_full[layout["start"] : layout["end"]]
        output_curve = _resample_curve(
            refined_segment,
            timestep,
            layout["source_timestep"],
            layout["source_length"],
        )
        segment["tension"] = _format_curve(output_curve)
        segment["tension_timestep"] = layout["source_timestep"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output_root, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"| Saved refined DS to {args.output}.")

    if args.diagnostics is not None:
        args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.diagnostics,
            timestep=np.float32(timestep),
            base_tension=base_tension,
            accompaniment_delta=delta_full,
            tension=refined_full,
            curve_mask=curve_mask,
        )
        print(f"| Saved full-song diagnostics to {args.diagnostics}.")


if __name__ == "__main__":
    main()
