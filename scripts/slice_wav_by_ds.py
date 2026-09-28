"""Slice a PCM WAV file into one file per segment in a DiffSinger DS file."""

from __future__ import annotations

import argparse
import json
import math
import wave
from pathlib import Path
from typing import Any


def parse_number_sequence(value: Any, field: str, index: int) -> list[float]:
    """Parse a DS whitespace-separated or JSON-array number sequence."""
    if isinstance(value, str):
        values = value.split()
    elif isinstance(value, list):
        values = value
    else:
        raise ValueError(
            f"Segment {index}: {field!r} must be a string or a JSON array."
        )

    try:
        numbers = [float(item) for item in values]
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Segment {index}: {field!r} contains a non-numeric value."
        ) from error

    if not numbers or any(not math.isfinite(number) or number < 0 for number in numbers):
        raise ValueError(
            f"Segment {index}: {field!r} must contain non-negative finite numbers."
        )
    return numbers


def load_segments(ds_path: Path, duration_field: str) -> list[tuple[int, float, float]]:
    """Return (original index, start seconds, end seconds) for every DS segment."""
    try:
        root = json.loads(ds_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {ds_path}: {error}") from error

    raw_segments = root if isinstance(root, list) else [root]
    if not raw_segments or not all(isinstance(item, dict) for item in raw_segments):
        raise ValueError("The DS root must be an object or a non-empty list of objects.")

    segments: list[tuple[int, float, float]] = []
    for index, segment in enumerate(raw_segments):
        try:
            offset = float(segment.get("offset", 0.0))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Segment {index}: invalid 'offset'.") from error
        if not math.isfinite(offset) or offset < 0:
            raise ValueError(f"Segment {index}: 'offset' must be non-negative.")

        if duration_field == "auto":
            field = "ph_dur" if segment.get("ph_dur") not in (None, "") else "note_dur"
        else:
            field = duration_field
        if segment.get(field) in (None, ""):
            raise ValueError(f"Segment {index}: missing {field!r}.")

        duration = sum(parse_number_sequence(segment[field], field, index))
        if duration <= 0:
            raise ValueError(f"Segment {index}: duration must be greater than zero.")
        segments.append((index, offset, offset + duration))

    return segments


def slice_wav(
    wav_path: Path,
    ds_path: Path,
    output_dir: Path,
    duration_field: str,
    overwrite: bool,
) -> list[Path]:
    segments = load_segments(ds_path, duration_field)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    with wave.open(str(wav_path), "rb") as source:
        if source.getcomptype() != "NONE":
            raise ValueError(
                f"Only uncompressed PCM WAV is supported; got {source.getcomptype()}."
            )

        params = source.getparams()
        frame_rate = source.getframerate()
        total_frames = source.getnframes()
        width = max(3, len(str(len(segments) - 1)))

        for index, start_seconds, end_seconds in segments:
            start_frame = round(start_seconds * frame_rate)
            end_frame = round(end_seconds * frame_rate)
            if start_frame >= total_frames:
                raise ValueError(
                    f"Segment {index} starts at {start_seconds:.6f}s, beyond the "
                    f"WAV duration ({total_frames / frame_rate:.6f}s)."
                )
            end_frame = min(end_frame, total_frames)
            if end_frame <= start_frame:
                raise ValueError(f"Segment {index} has no audio frames to write.")

            output_path = output_dir / f"{index}.wav"
            if output_path.exists() and not overwrite:
                raise FileExistsError(
                    f"Output already exists: {output_path}. Use --overwrite to replace it."
                )

            source.setpos(start_frame)
            audio_frames = source.readframes(end_frame - start_frame)
            with wave.open(str(output_path), "wb") as target:
                target.setparams(params)
                target.writeframes(audio_frames)
            written.append(output_path)

            actual_end = end_frame / frame_rate
            print(
                f"[{index:0{width}d}] {start_frame / frame_rate:.6f}s - "
                f"{actual_end:.6f}s -> {output_path}"
            )

    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Slice an uncompressed PCM WAV into one file per DiffSinger DS segment. "
            "Each slice starts at 'offset' and lasts for the sum of its duration field."
        )
    )
    parser.add_argument("wav", type=Path, help="Source WAV file")
    parser.add_argument("ds", type=Path, help="DiffSinger .ds file")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Output directory (default: <wav_stem>_slices beside the WAV)",
    )
    parser.add_argument(
        "--duration-field",
        choices=("auto", "ph_dur", "note_dur"),
        default="auto",
        help="Duration sequence to sum (default: ph_dur, falling back to note_dur)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output WAV files that already exist",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    wav_path = args.wav.expanduser().resolve()
    ds_path = args.ds.expanduser().resolve()
    if not wav_path.is_file():
        raise SystemExit(f"WAV file does not exist: {wav_path}")
    if not ds_path.is_file():
        raise SystemExit(f"DS file does not exist: {ds_path}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else wav_path.with_name(f"{wav_path.stem}_slices")
    )
    try:
        written = slice_wav(
            wav_path,
            ds_path,
            output_dir,
            args.duration_field,
            args.overwrite,
        )
    except (FileExistsError, OSError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
    print(f"Wrote {len(written)} slice(s) to {output_dir}")


if __name__ == "__main__":
    main()
