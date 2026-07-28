"""Visualize generated DiffSinger variance curves stored in a .ds file."""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from .ds_parser import load_generated_variances
from .ds_render import write_generated_variance_visualization


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create an interactive timeline for generated pitch, energy, "
            "breathiness, voicing, and tension curves in a DiffSinger DS file."
        )
    )
    parser.add_argument("ds", type=Path, help="Input .ds file")
    parser.add_argument(
        "--audio",
        type=Path,
        help="Generated vocal WAV; defaults to a same-stem WAV beside the DS file",
    )
    parser.add_argument(
        "--accompaniment",
        type=Path,
        help="Optional accompaniment WAV synchronized to --audio",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output HTML path (default: <ds-stem>-generated-variance.html)",
    )
    audio_group = parser.add_mutually_exclusive_group()
    audio_group.add_argument(
        "--embed-audio",
        dest="embed_audio",
        action="store_true",
        default=True,
        help="Embed audio in the HTML (default; portable between machines)",
    )
    audio_group.add_argument(
        "--link-audio",
        dest="embed_audio",
        action="store_false",
        help="Reference local audio paths instead of embedding them",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Create the visualization without opening a browser",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ds_path = args.ds.expanduser().resolve()
    audio_path = args.audio.expanduser().resolve() if args.audio else None
    if audio_path is None:
        candidate = ds_path.with_suffix(".wav")
        if candidate.is_file():
            audio_path = candidate
    accompaniment_path = (
        args.accompaniment.expanduser().resolve()
        if args.accompaniment
        else None
    )
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else ds_path.with_name(f"{ds_path.stem}-generated-variance.html")
    )

    try:
        data = load_generated_variances(ds_path)
        result = write_generated_variance_visualization(
            data,
            output_path,
            audio_path=audio_path,
            accompaniment_path=accompaniment_path,
            embed_audio=args.embed_audio,
        )
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Created generated-variance visualization: {result}")
    if audio_path is None:
        print("Note: no same-stem WAV found; pass --audio to enable playback.")
    if not args.no_open:
        webbrowser.open(result.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())