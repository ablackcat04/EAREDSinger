from __future__ import annotations

import base64
import html
import json
import mimetypes
import wave
from pathlib import Path

from .ds_parser import GeneratedVarianceData


def _audio_source(path: Path, embed_audio: bool) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {path}")
    if not embed_audio:
        return path.resolve().as_uri()
    mime_type = mimetypes.guess_type(path.name)[0] or "audio/wav"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def _wav_duration(path: Path | None) -> float:
    if path is None or path.suffix.lower() != ".wav":
        return 0.0
    try:
        with wave.open(str(path), "rb") as stream:
            return stream.getnframes() / stream.getframerate()
    except (wave.Error, OSError, ZeroDivisionError):
        return 0.0


def write_generated_variance_visualization(
    data: GeneratedVarianceData,
    output_path: str | Path,
    *,
    audio_path: str | Path | None = None,
    accompaniment_path: str | Path | None = None,
    embed_audio: bool = True,
) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vocal_path = Path(audio_path).expanduser().resolve() if audio_path else None
    accompaniment = (
        Path(accompaniment_path).expanduser().resolve()
        if accompaniment_path
        else None
    )
    if accompaniment is not None and vocal_path is None:
        raise ValueError(
            "An accompaniment requires --audio to drive synchronized playback."
        )

    duration = max(data.duration, _wav_duration(vocal_path))
    payload = {
        "name": data.source_path.name,
        "duration": round(duration, 6),
        "curves": [
            {
                "id": curve.id,
                "label": curve.label,
                "unit": curve.unit,
                "times": [round(value, 6) for value in curve.times],
                "values": [
                    round(value, 5) if value is not None else None
                    for value in curve.values
                ],
            }
            for curve in data.curves
        ],
        "segments": [
            {
                "offset": round(segment.offset, 6),
                "end": round(segment.end, 6),
                "text": segment.text,
            }
            for segment in data.segments
        ],
    }
    data_json = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    ).replace("</", "<\\/")

    vocal_element = ""
    accompaniment_element = ""
    accompaniment_control = ""
    if vocal_path is not None:
        vocal_src = html.escape(_audio_source(vocal_path, embed_audio), quote=True)
        vocal_element = (
            f'<audio id="audio" controls preload="metadata" '
            f'src="{vocal_src}"></audio>'
        )
    if accompaniment is not None:
        accompaniment_src = html.escape(
            _audio_source(accompaniment, embed_audio), quote=True
        )
        accompaniment_element = (
            f'<audio id="accompaniment" preload="auto" '
            f'src="{accompaniment_src}"></audio>'
        )
        accompaniment_control = (
            '<label>Accompaniment '
            '<input id="accompaniment-volume" type="range" min="0" max="1" '
            'step="0.01" value="0.7"></label>'
        )
    playback_message = (
        ""
        if vocal_path is not None
        else '<span class="muted">No audio supplied; use --audio for playback.</span>'
    )

    template_path = Path(__file__).with_name("generated_variance_template.html")
    template = template_path.read_text(encoding="utf-8")
    replacements = {
        "__TITLE__": html.escape(
            f"{data.source_path.stem} — generated variance"
        ),
        "__NAME__": html.escape(data.source_path.name),
        "__DATA__": data_json,
        "__VOCAL_AUDIO__": vocal_element,
        "__ACCOMPANIMENT_AUDIO__": accompaniment_element,
        "__ACCOMPANIMENT_CONTROL__": accompaniment_control,
        "__PLAYBACK_MESSAGE__": playback_message,
    }
    for marker, value in replacements.items():
        template = template.replace(marker, value)

    output_path.write_text(template, encoding="utf-8")
    return output_path
