from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GeneratedCurve:
    id: str
    label: str
    unit: str
    times: tuple[float, ...]
    values: tuple[float | None, ...]


@dataclass(frozen=True)
class DSSegment:
    offset: float
    end: float
    text: str


@dataclass(frozen=True)
class GeneratedVarianceData:
    source_path: Path
    duration: float
    curves: tuple[GeneratedCurve, ...]
    segments: tuple[DSSegment, ...]


CURVE_SPECS = (
    ("f0_seq", "pitch", "Generated pitch", "Hz"),
    ("energy", "energy", "Generated energy", "dB"),
    ("breathiness", "breathiness", "Generated breathiness", "dB"),
    ("voicing", "voicing", "Generated voicing", "dB"),
    ("tension", "tension", "Generated tension", "logit"),
)


def _float_sequence(value: object, field: str, segment_index: int) -> list[float]:
    if isinstance(value, str):
        tokens = value.split()
    elif isinstance(value, list):
        tokens = value
    else:
        raise ValueError(
            f"Segment {segment_index}: {field!r} must be a "
            "whitespace-separated string or a list."
        )
    try:
        return [float(token) for token in tokens]
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Segment {segment_index}: {field!r} contains a non-numeric value."
        ) from error


def _curve_timestep(segment: dict, field: str, segment_index: int) -> float:
    key = "f0_timestep" if field == "f0_seq" else f"{field}_timestep"
    if key not in segment:
        raise ValueError(
            f"Segment {segment_index}: {field!r} exists without {key!r}."
        )
    try:
        timestep = float(segment[key])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Segment {segment_index}: invalid {key!r} value."
        ) from error
    if not math.isfinite(timestep) or timestep <= 0:
        raise ValueError(f"Segment {segment_index}: {key!r} must be positive.")
    return timestep


def load_generated_variances(path: str | Path) -> GeneratedVarianceData:
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"DS file does not exist: {source_path}")
    try:
        root = json.loads(source_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in DS file: {error}") from error

    raw_segments = root if isinstance(root, list) else [root]
    if not raw_segments or not all(isinstance(item, dict) for item in raw_segments):
        raise ValueError(
            "The DS root must be an object or a non-empty list of objects."
        )
    indexed_segments = sorted(
        enumerate(raw_segments),
        key=lambda pair: float(pair[1].get("offset", 0.0)),
    )
    buffers = {
        field: {"times": [], "values": []}
        for field, _, _, _ in CURVE_SPECS
    }
    segments: list[DSSegment] = []
    duration = 0.0

    for sorted_index, (original_index, segment) in enumerate(indexed_segments):
        try:
            offset = float(segment.get("offset", 0.0))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Segment {original_index}: invalid offset."
            ) from error
        if not math.isfinite(offset) or offset < 0:
            raise ValueError(
                f"Segment {original_index}: offset must be non-negative."
            )
        next_offset = (
            float(indexed_segments[sorted_index + 1][1].get("offset", 0.0))
            if sorted_index + 1 < len(indexed_segments)
            else None
        )

        segment_end = offset
        if segment.get("note_dur"):
            segment_end = max(
                segment_end,
                offset
                + sum(
                    _float_sequence(
                        segment["note_dur"], "note_dur", original_index
                    )
                ),
            )

        for field, _, _, _ in CURVE_SPECS:
            if field not in segment or segment[field] in (None, ""):
                continue
            values = _float_sequence(segment[field], field, original_index)
            if not values:
                continue
            timestep = _curve_timestep(segment, field, original_index)
            buffer = buffers[field]
            if buffer["values"]:
                buffer["times"].append(offset)
                buffer["values"].append(None)
            for value_index, value in enumerate(values):
                sample_time = offset + value_index * timestep
                if next_offset is not None and sample_time >= next_offset:
                    break
                buffer["times"].append(sample_time)
                buffer["values"].append(value if math.isfinite(value) else None)
            segment_end = max(segment_end, offset + len(values) * timestep)

        if next_offset is not None:
            segment_end = min(segment_end, next_offset)
        duration = max(duration, segment_end)
        segments.append(
            DSSegment(
                offset=offset,
                end=segment_end,
                text=str(segment.get("text", "")).strip(),
            )
        )

    curves = []
    for field, curve_id, label, unit in CURVE_SPECS:
        buffer = buffers[field]
        if buffer["values"]:
            curves.append(
                GeneratedCurve(
                    id=curve_id,
                    label=label,
                    unit=unit,
                    times=tuple(buffer["times"]),
                    values=tuple(buffer["values"]),
                )
            )
    if not curves:
        supported = ", ".join(field for field, _, _, _ in CURVE_SPECS)
        raise ValueError(
            f"No generated curves found. Supported fields: {supported}."
        )

    return GeneratedVarianceData(
        source_path=source_path,
        duration=duration,
        curves=tuple(curves),
        segments=tuple(segments),
    )
