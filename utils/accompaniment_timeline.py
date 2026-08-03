from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SegmentTiming:
    name: str
    start: float
    end: float


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    for encoding in ("utf-8-sig", "utf-8", "utf-16"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


def _parse_interval_tiers(text: str) -> dict[str, list[tuple[float, float, str]]]:
    tiers: dict[str, list[tuple[float, float, str]]] = {}
    chunks = re.split(r"item\s*\[\d+\]\s*:", text)[1:]
    interval_pattern = re.compile(
        r"intervals\s*\[\d+\]\s*:\s*"
        r"xmin\s*=\s*([-+0-9.eE]+)\s*"
        r"xmax\s*=\s*([-+0-9.eE]+)\s*"
        r'text\s*=\s*"((?:[^"]|"")*)"',
        re.DOTALL,
    )
    for chunk in chunks:
        class_match = re.search(r'class\s*=\s*"([^"]+)"', chunk)
        name_match = re.search(r'name\s*=\s*"([^"]+)"', chunk)
        if class_match is None or name_match is None:
            continue
        if class_match.group(1) != "IntervalTier":
            continue
        intervals = [
            (float(match.group(1)), float(match.group(2)), match.group(3).replace('""', '"').strip())
            for match in interval_pattern.finditer(chunk)
        ]
        tiers[name_match.group(1)] = intervals
    return tiers


def load_song_segment_timings(
    textgrid_path: str | Path,
    *,
    sentence_tier: str = "句子",
    silence_label: str = "silence",
) -> dict[str, SegmentTiming]:
    """Map ``<song>_NNN`` names to their original full-song time ranges."""
    path = Path(textgrid_path)
    tiers = _parse_interval_tiers(_read_text(path))
    if sentence_tier not in tiers:
        available = ", ".join(repr(name) for name in tiers)
        raise ValueError(
            f"Sentence tier {sentence_tier!r} not found in {path}. "
            f"Available interval tiers: {available}"
        )

    song_id = path.stem
    timings: dict[str, SegmentTiming] = {}
    segment_index = 0
    for start, end, label in tiers[sentence_tier]:
        if not label or label.casefold() == silence_label.casefold():
            continue
        if end <= start:
            raise ValueError(f"Invalid sentence interval {start}..{end} in {path}.")
        segment_index += 1
        name = f"{song_id}_{segment_index:03d}"
        timings[name] = SegmentTiming(name=name, start=start, end=end)
    return timings


def strip_dataset_prefix(item_name: str | bytes) -> str:
    if isinstance(item_name, bytes):
        item_name = item_name.decode("utf-8")
    return item_name.split(":", maxsplit=1)[-1]


def song_id_from_segment_name(segment_name: str) -> str:
    song_id, separator, segment_index = segment_name.rpartition("_")
    if not separator or not song_id or not segment_index.isdigit():
        raise ValueError(
            f"Expected a segment name like '2001_001', received {segment_name!r}."
        )
    return song_id
