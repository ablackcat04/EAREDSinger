from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from .extractor import ExtractionConfig, extract_variances
from .render import write_interactive_visualization


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract DiffSinger pitch/energy/breathiness/voicing/tension from a "
            "vocal WAV and create an interactive, playable timeline."
        )
    )
    parser.add_argument("audio", type=Path, help="Input WAV file")
    parser.add_argument(
        "--accompaniment",
        type=Path,
        help="Optional accompaniment WAV to play in sync with the vocal",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output HTML path (default: <audio-name>-variance.html beside the WAV)",
    )
    parser.add_argument(
        "--separator",
        choices=("world", "vr"),
        default="world",
        help="DiffSinger harmonic/noise separator (default: world)",
    )
    parser.add_argument(
        "--vr-checkpoint",
        type=Path,
        help="Vocal Remover model checkpoint required by --separator vr",
    )
    parser.add_argument("--device", default="cpu", help="Torch device, such as cpu or cuda")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--hop-size", type=int, default=512)
    parser.add_argument("--win-size", type=int, default=2048)
    parser.add_argument("--fft-size", type=int, default=2048)
    parser.add_argument("--f0-min", type=float, default=65.0)
    parser.add_argument("--f0-max", type=float, default=1100.0)
    parser.add_argument("--smooth-width", type=float, default=0.12, help="Smoothing width in seconds")
    audio_group = parser.add_mutually_exclusive_group()
    audio_group.add_argument(
        "--embed-audio",
        dest="embed_audio",
        action="store_true",
        default=True,
        help="Embed the WAV in the HTML (default; portable between machines)",
    )
    audio_group.add_argument(
        "--link-audio",
        dest="embed_audio",
        action="store_false",
        help=(
            "Reference the server-side audio path instead of embedding it; "
            "the result will not be portable to another machine"
        ),
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Create the visualization without opening it in the default browser",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audio_path = args.audio.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else audio_path.with_name(f"{audio_path.stem}-variance.html")
    )
    config = ExtractionConfig(
        sample_rate=args.sample_rate,
        hop_size=args.hop_size,
        win_size=args.win_size,
        fft_size=args.fft_size,
        f0_min=args.f0_min,
        f0_max=args.f0_max,
        smooth_width=args.smooth_width,
        separator=args.separator,
        vr_checkpoint=args.vr_checkpoint,
        device=args.device,
    )

    try:
        print(f"Extracting DiffSinger variance features from {audio_path} ...")
        features = extract_variances(audio_path, config)
        result = write_interactive_visualization(
            features,
            output_path,
            embed_audio=args.embed_audio,
            accompaniment_path=args.accompaniment,
            analysis_win_size=args.win_size,
            analysis_fft_size=args.fft_size,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Created interactive visualization: {result}")
    if not args.no_open:
        webbrowser.open(result.as_uri())
    return 0
