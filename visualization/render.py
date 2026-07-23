from __future__ import annotations

import base64
import html
import json
import mimetypes
from pathlib import Path

import librosa
import numpy as np

from .extractor import VarianceFeatures


def _finite_or_none(values: np.ndarray) -> list[float | None]:
    return [
        round(float(value), 5) if np.isfinite(value) else None
        for value in values
    ]


def _raw_waveform_envelope(
    waveform: np.ndarray, hop_size: int
) -> tuple[np.ndarray, np.ndarray]:
    frame_count = max(1, (len(waveform) + hop_size - 1) // hop_size)
    required = frame_count * hop_size
    padded = np.pad(waveform, (0, required - len(waveform)))
    frames = padded.reshape(frame_count, hop_size)
    return frames.min(axis=1), frames.max(axis=1)


def _waveform_envelope(features: VarianceFeatures) -> tuple[list[float], list[float]]:
    minimum, maximum = _raw_waveform_envelope(features.waveform, features.hop_size)
    return _finite_or_none(minimum), _finite_or_none(maximum)


def _reduce_spectrogram(
    spectrum: np.ndarray, times: np.ndarray, frequencies: np.ndarray,
    *, max_time_bins: int = 1800, max_frequency_bins: int = 96
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if spectrum.shape[1] > max_time_bins:
        edges = np.linspace(0, spectrum.shape[1], max_time_bins + 1, dtype=np.int64)
        spectrum = np.stack([
            spectrum[:, edges[i]:edges[i + 1]].max(axis=1)
            for i in range(max_time_bins)
        ], axis=1)
        times = np.array([
            times[edges[i]:edges[i + 1]].mean()
            for i in range(max_time_bins)
        ])
    if spectrum.shape[0] > max_frequency_bins:
        edges = np.linspace(0, spectrum.shape[0], max_frequency_bins + 1, dtype=np.int64)
        spectrum = np.stack([
            spectrum[edges[i]:edges[i + 1]].max(axis=0)
            for i in range(max_frequency_bins)
        ])
        frequencies = np.array([
            frequencies[edges[i]:edges[i + 1]].mean()
            for i in range(max_frequency_bins)
        ])
    return spectrum, times, frequencies


def _analyze_accompaniment(
    path: Path, sample_rate: int, hop_size: int, win_size: int, fft_size: int
) -> dict:
    waveform, _ = librosa.load(path, sr=sample_rate, mono=True, dtype=np.float32)
    if waveform.size == 0:
        raise ValueError(f"Accompaniment file is empty: {path}")
    wave_min, wave_max = _raw_waveform_envelope(waveform, hop_size)
    times = np.arange(len(wave_min), dtype=np.float64) * hop_size / sample_rate
    loudness = librosa.feature.rms(
        y=waveform, frame_length=win_size, hop_length=hop_size
    )[0]
    if len(loudness) < len(times):
        loudness = np.pad(loudness, (0, len(times) - len(loudness)))
    loudness = librosa.amplitude_to_db(loudness[:len(times)]).astype(np.float32)

    magnitude = np.abs(librosa.stft(
        waveform, n_fft=fft_size, hop_length=hop_size, win_length=win_size
    ))
    spectrum = librosa.amplitude_to_db(
        magnitude, ref=np.max, top_db=96.0
    ).astype(np.float32)
    spectrum_times = librosa.frames_to_time(
        np.arange(spectrum.shape[1]), sr=sample_rate, hop_length=hop_size
    )
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=fft_size)
    spectrum, spectrum_times, frequencies = _reduce_spectrogram(
        spectrum, spectrum_times, frequencies
    )
    spectrum = np.rint(spectrum).astype(np.int16)

    return {
        "name": path.name,
        "times": _finite_or_none(times),
        "waveMin": _finite_or_none(wave_min),
        "waveMax": _finite_or_none(wave_max),
        "loudness": _finite_or_none(loudness),
        "spectrumTimes": _finite_or_none(spectrum_times),
        "frequencies": _finite_or_none(frequencies),
        "spectrum": spectrum.tolist(),
        "spectrumMin": int(spectrum.min()),
        "spectrumMax": int(spectrum.max()),
    }


def _audio_source(path: Path, embed_audio: bool) -> str:
    if not embed_audio:
        return path.resolve().as_uri()
    mime_type = mimetypes.guess_type(path.name)[0] or "audio/wav"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def write_interactive_visualization(
    features: VarianceFeatures,
    output_path: str | Path,
    *,
    embed_audio: bool = True,
    accompaniment_path: str | Path | None = None,
    analysis_win_size: int = 2048,
    analysis_fft_size: int = 2048,
) -> Path:
    """Write a self-contained interactive timeline, optionally embedding its audio."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wave_min, wave_max = _waveform_envelope(features)
    pitch = features.pitch_midi.astype(np.float64, copy=True)
    pitch[features.unvoiced] = np.nan
    resolved_accompaniment = (
        Path(accompaniment_path).expanduser().resolve()
        if accompaniment_path is not None
        else None
    )
    accompaniment_data = (
        _analyze_accompaniment(
            resolved_accompaniment, features.sample_rate, features.hop_size,
            analysis_win_size, analysis_fft_size
        )
        if resolved_accompaniment is not None
        else None
    )

    payload = {
        "name": features.source_path.name,
        "duration": round(features.duration, 6),
        "timestep": round(features.hop_size / features.sample_rate, 9),
        "times": _finite_or_none(features.times),
        "waveMin": wave_min,
        "waveMax": wave_max,
        "accompaniment": accompaniment_data,
        "tracks": [
            {
                "id": "pitch",
                "label": "Pitch",
                "unit": "MIDI",
                "series": [
                    {"label": "Pitch", "values": _finite_or_none(pitch), "style": 0}
                ],
            },
            {
                "id": "energy",
                "label": "Energy",
                "unit": "dB",
                "series": [
                    {"label": "Energy", "values": _finite_or_none(features.energy), "style": 0}
                ],
            },
            {
                "id": "components",
                "label": "Harmonic / aperiodic energy",
                "unit": "dB",
                "series": [
                    {"label": "Voicing", "values": _finite_or_none(features.voicing), "style": 0},
                    {
                        "label": "Breathiness",
                        "values": _finite_or_none(features.breathiness),
                        "style": 1,
                    },
                ],
            },
            {
                "id": "tension",
                "label": "Tension",
                "unit": "logit",
                "series": [
                    {"label": "Tension", "values": _finite_or_none(features.tension), "style": 0}
                ],
            },
        ],
    }
    if accompaniment_data is not None:
        payload["tracks"].append({
            "id": "accompaniment-loudness",
            "label": "Accompaniment short-time loudness",
            "unit": "dB",
            "series": [{
                "label": "Loudness",
                "values": accompaniment_data["loudness"],
                "style": 0,
            }],
            "times": accompaniment_data["times"],
        })

    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    audio_src = html.escape(_audio_source(features.source_path, embed_audio), quote=True)
    accompaniment_src = (
        html.escape(
            _audio_source(resolved_accompaniment, embed_audio),
            quote=True,
        )
        if accompaniment_path is not None
        else None
    )
    accompaniment_element = (
        f'<audio id="accompaniment" preload="auto" src="{accompaniment_src}"></audio>'
        if accompaniment_src is not None
        else ""
    )
    accompaniment_control = (
        '<label>Accompaniment '
        '<input id="accompaniment-volume" type="range" min="0" max="1" '
        'step="0.01" value="0.7"></label>'
        if accompaniment_src is not None
        else ""
    )
    title = html.escape(f"{features.source_path.stem} — DiffSinger variance")

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{
    color-scheme: light dark;
    --bg: #f6f7fb;
    --panel: #ffffff;
    --text: #18212f;
    --muted: #607086;
    --border: #d8dee8;
    --grid: #e5e9f0;
    --series-1: #246bfd;
    --series-2: #d64c7f;
    --cursor: #f29d38;
    --shade: rgba(36, 107, 253, 0.10);
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #10141b;
      --panel: #171d27;
      --text: #edf2f8;
      --muted: #a8b4c5;
      --border: #303a49;
      --grid: #293240;
      --series-1: #72a4ff;
      --series-2: #ff83b1;
      --cursor: #ffc46b;
      --shade: rgba(114, 164, 255, 0.12);
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
  }}
  main {{ width: min(1440px, 100%); margin: 0 auto; padding: 18px; }}
  header {{ display: flex; gap: 16px; align-items: baseline; flex-wrap: wrap; }}
  h1 {{ margin: 0; font-size: 20px; font-weight: 650; }}
  .meta {{ color: var(--muted); }}
  .transport {{
    position: sticky;
    top: 0;
    z-index: 3;
    display: grid;
    grid-template-columns: minmax(260px, 1fr) auto;
    gap: 12px;
    align-items: center;
    margin: 14px 0;
    padding: 12px;
    background: color-mix(in srgb, var(--panel) 94%, transparent);
    border: 1px solid var(--border);
    border-radius: 10px;
    backdrop-filter: blur(10px);
  }}
  audio {{ width: 100%; min-width: 0; }}
  .controls {{ display: flex; align-items: center; gap: 7px; flex-wrap: wrap; }}
  button {{
    border: 1px solid var(--border);
    border-radius: 7px;
    background: var(--panel);
    color: var(--text);
    min-height: 34px;
    padding: 5px 10px;
    cursor: pointer;
  }}
  button:hover {{ border-color: var(--series-1); }}
  button:focus-visible {{ outline: 2px solid var(--series-1); outline-offset: 2px; }}
  label {{ white-space: nowrap; color: var(--muted); }}
  .timeline {{
    display: grid;
    gap: 10px;
    user-select: none;
  }}
  .track {{
    position: relative;
    overflow: hidden;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 9px;
  }}
  .track-head {{
    display: flex;
    justify-content: space-between;
    gap: 12px;
    padding: 8px 11px 0;
    pointer-events: none;
  }}
  .track-title {{ font-weight: 650; }}
  .track-value {{ color: var(--muted); font-variant-numeric: tabular-nums; }}
  canvas {{
    display: block;
    width: 100%;
    height: 136px;
    cursor: crosshair;
    touch-action: none;
  }}
  .wave canvas {{ height: 102px; }}
  .spectrum canvas {{ height: 210px; }}
  .axis-input {{ width: 78px; min-height: 32px; }}
  select {{ min-height: 32px; max-width: 180px; }}
  .help {{ margin: 10px 2px 0; color: var(--muted); }}
  @media (max-width: 720px) {{
    main {{ padding: 10px; }}
    .transport {{ grid-template-columns: 1fr; position: static; }}
    canvas {{ height: 120px; }}
  }}
</style>
</head>
<body>
<main>
  <header>
    <h1>{html.escape(features.source_path.name)}</h1>
    <span class="meta" id="metadata"></span>
  </header>
  <section class="transport" aria-label="Playback and timeline controls">
    <audio id="audio" controls preload="metadata" src="{audio_src}"></audio>
    {accompaniment_element}
    <div class="controls">
      <button id="zoom-in" type="button" aria-label="Zoom in timeline">Zoom in</button>
      <button id="zoom-out" type="button" aria-label="Zoom out timeline">Zoom out</button>
      <button id="full-view" type="button">Full view</button>
      <label><input id="follow" type="checkbox" checked> Follow playback</label>
      {accompaniment_control}
      <label>Y axis <select id="axis-track"></select></label>
      <label>Min <input id="axis-min" class="axis-input" type="number" step="any"></label>
      <label>Max <input id="axis-max" class="axis-input" type="number" step="any"></label>
      <button id="axis-apply" type="button">Apply Y</button>
      <button id="axis-auto" type="button">Auto Y</button>
    </div>
  </section>
  <section class="timeline" id="timeline" aria-label="Extracted variance timeline">
    <div class="track wave">
      <div class="track-head"><span class="track-title">Waveform</span><span class="track-value" data-value="wave"></span></div>
      <canvas data-track="wave" data-label="Vocal waveform" aria-label="Vocal waveform timeline"></canvas>
    </div>
  </section>
  <p class="help">Scroll over a plot or use the buttons to zoom. Drag to pan. Click to seek playback.</p>
</main>
<script>
const DATA = {data_json};
const root = document.getElementById("timeline");
const audio = document.getElementById("audio");
const accompaniment = document.getElementById("accompaniment");
const accompanimentVolume = document.getElementById("accompaniment-volume");
const axisTrack = document.getElementById("axis-track");
const axisMin = document.getElementById("axis-min");
const axisMax = document.getElementById("axis-max");
const css = getComputedStyle(document.documentElement);
const colors = {{
  text: css.getPropertyValue("--text").trim(),
  muted: css.getPropertyValue("--muted").trim(),
  grid: css.getPropertyValue("--grid").trim(),
  one: css.getPropertyValue("--series-1").trim(),
  two: css.getPropertyValue("--series-2").trim(),
  cursor: css.getPropertyValue("--cursor").trim(),
  shade: css.getPropertyValue("--shade").trim()
}};
const state = {{
  start: 0,
  end: DATA.duration,
  hoverTime: null,
  dragging: false,
  dragStartX: 0,
  dragStartRange: null,
  moved: false,
  yRanges: {{}}
}};
const margins = {{ left: 58, right: 14, top: 10, bottom: 25 }};
const canvases = [];

if (DATA.accompaniment) {{
  const waveSection = document.createElement("div");
  waveSection.className = "track wave";
  waveSection.innerHTML = `<div class="track-head"><span class="track-title">Accompaniment waveform</span><span class="track-value" data-value="accompaniment-wave"></span></div><canvas data-track="accompaniment-wave" data-label="Accompaniment waveform" aria-label="Accompaniment waveform timeline"></canvas>`;
  root.appendChild(waveSection);
  const spectrumSection = document.createElement("div");
  spectrumSection.className = "track spectrum";
  spectrumSection.innerHTML = `<div class="track-head"><span class="track-title">Accompaniment spectrogram</span><span class="track-value" data-value="accompaniment-spectrum">relative dB</span></div><canvas data-track="accompaniment-spectrum" data-label="Accompaniment spectrogram" aria-label="Accompaniment spectrogram in hertz"></canvas>`;
  root.appendChild(spectrumSection);
}}

for (const track of DATA.tracks) {{
  const section = document.createElement("div");
  section.className = "track";
  const legend = track.series.map((s, i) =>
    `<span><span style="color:${{i ? colors.two : colors.one}}">●</span> ${{s.label}}</span>`
  ).join(" &nbsp; ");
  section.innerHTML = `<div class="track-head"><span class="track-title">${{track.label}} <span class="meta">${{legend}}</span></span><span class="track-value" data-value="${{track.id}}"></span></div><canvas data-track="${{track.id}}" data-label="${{track.label}}" aria-label="${{track.label}} in ${{track.unit}}"></canvas>`;
  root.appendChild(section);
}}

for (const canvas of root.querySelectorAll("canvas")) {{
  canvases.push(canvas);
  const option = document.createElement("option");
  option.value = canvas.dataset.track;
  option.textContent = canvas.dataset.label || canvas.dataset.track;
  axisTrack.appendChild(option);
  canvas.addEventListener("wheel", onWheel, {{ passive: false }});
  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("pointercancel", onPointerUp);
  canvas.addEventListener("pointerleave", () => {{
    if (!state.dragging) {{
      state.hoverTime = null;
      updateReadouts();
      drawAll();
    }}
  }});
}}

function clamp(value, low, high) {{ return Math.max(low, Math.min(high, value)); }}
function visibleDuration() {{ return state.end - state.start; }}
function plotWidth(canvas) {{ return canvas.clientWidth - margins.left - margins.right; }}
function xToTime(canvas, x) {{
  return state.start + clamp((x - margins.left) / plotWidth(canvas), 0, 1) * visibleDuration();
}}
function timeToX(canvas, time) {{
  return margins.left + (time - state.start) / visibleDuration() * plotWidth(canvas);
}}
function setRange(start, end) {{
  const minimum = Math.min(0.10, DATA.duration);
  let span = clamp(end - start, minimum, DATA.duration);
  let nextStart = start;
  if (nextStart < 0) nextStart = 0;
  if (nextStart + span > DATA.duration) nextStart = DATA.duration - span;
  state.start = Math.max(0, nextStart);
  state.end = Math.min(DATA.duration, state.start + span);
  drawAll();
}}
function zoom(factor, anchor = (state.start + state.end) / 2) {{
  const span = visibleDuration();
  const next = clamp(span * factor, Math.min(0.10, DATA.duration), DATA.duration);
  const ratio = span > 0 ? (anchor - state.start) / span : 0.5;
  setRange(anchor - next * ratio, anchor + next * (1 - ratio));
}}
function onWheel(event) {{
  event.preventDefault();
  const rect = event.currentTarget.getBoundingClientRect();
  const anchor = xToTime(event.currentTarget, event.clientX - rect.left);
  zoom(event.deltaY < 0 ? 0.75 : 1.34, anchor);
}}
function onPointerDown(event) {{
  const canvas = event.currentTarget;
  canvas.setPointerCapture(event.pointerId);
  state.dragging = true;
  state.moved = false;
  state.dragStartX = event.clientX;
  state.dragStartRange = [state.start, state.end];
}}
function onPointerMove(event) {{
  const canvas = event.currentTarget;
  const rect = canvas.getBoundingClientRect();
  state.hoverTime = xToTime(canvas, event.clientX - rect.left);
  if (state.dragging) {{
    const dx = event.clientX - state.dragStartX;
    if (Math.abs(dx) > 3) state.moved = true;
    const span = state.dragStartRange[1] - state.dragStartRange[0];
    const shift = -dx / plotWidth(canvas) * span;
    setRange(state.dragStartRange[0] + shift, state.dragStartRange[1] + shift);
  }} else {{
    updateReadouts();
    drawAll();
  }}
}}
function onPointerUp(event) {{
  if (!state.dragging) return;
  const canvas = event.currentTarget;
  if (!state.moved) {{
    const rect = canvas.getBoundingClientRect();
    audio.currentTime = xToTime(canvas, event.clientX - rect.left);
  }}
  state.dragging = false;
  state.dragStartRange = null;
}}

function indexAt(time) {{
  return clamp(Math.round(time / DATA.timestep), 0, DATA.times.length - 1);
}}
function formatTime(seconds) {{
  const minutes = Math.floor(seconds / 60);
  const secs = seconds - minutes * 60;
  return `${{minutes}}:${{secs.toFixed(2).padStart(5, "0")}}`;
}}
function niceTicks(min, max, count = 4) {{
  const raw = Math.max((max - min) / count, Number.EPSILON);
  const power = Math.pow(10, Math.floor(Math.log10(raw)));
  const normalized = raw / power;
  const step = (normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10) * power;
  const values = [];
  for (let value = Math.ceil(min / step) * step; value <= max + step * 0.1; value += step) values.push(value);
  return values;
}}
function lowerBound(values, target) {{
  let low = 0, high = values.length;
  while (low < high) {{
    const middle = (low + high) >> 1;
    if (values[middle] < target) low = middle + 1; else high = middle;
  }}
  return low;
}}
function visibleIndices(times = DATA.times) {{
  const first = clamp(lowerBound(times, state.start) - 1, 0, times.length - 1);
  const last = clamp(lowerBound(times, state.end) + 1, 0, times.length - 1);
  return [first, last];
}}
function axisBounds(id, autoLow, autoHigh) {{
  const range = state.yRanges[id];
  if (range && Number.isFinite(range[0]) && Number.isFinite(range[1]) && range[1] > range[0]) return range;
  if (axisTrack.value === id) {{
    axisMin.placeholder = autoLow.toFixed(Math.abs(autoLow) < 10 ? 2 : 1);
    axisMax.placeholder = autoHigh.toFixed(Math.abs(autoHigh) < 10 ? 2 : 1);
  }}
  return [autoLow, autoHigh];
}}
function trackFor(id) {{ return DATA.tracks.find(track => track.id === id); }}
function boundsFor(track, first, last) {{
  let low = Infinity, high = -Infinity;
  for (const series of track.series) {{
    for (let i = first; i <= last; i++) {{
      const value = series.values[i];
      if (value !== null && Number.isFinite(value)) {{
        low = Math.min(low, value);
        high = Math.max(high, value);
      }}
    }}
  }}
  if (!Number.isFinite(low)) return [0, 1];
  if (low === high) return [low - 1, high + 1];
  const pad = (high - low) * 0.08;
  return [low - pad, high + pad];
}}
function prepare(canvas) {{
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {{
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
  }}
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return [ctx, width, height];
}}
function drawAxes(ctx, canvas, height, low, high, unit) {{
  const bottom = height - margins.bottom;
  ctx.font = "12px system-ui";
  ctx.lineWidth = 1;
  ctx.strokeStyle = colors.grid;
  ctx.fillStyle = colors.muted;
  ctx.textBaseline = "middle";
  for (const value of niceTicks(low, high, 3)) {{
    const y = margins.top + (high - value) / (high - low) * (bottom - margins.top);
    ctx.beginPath(); ctx.moveTo(margins.left, y); ctx.lineTo(canvas.clientWidth - margins.right, y); ctx.stroke();
    ctx.textAlign = "right";
    ctx.fillText(value.toFixed(Math.abs(value) < 10 ? 1 : 0), margins.left - 7, y);
  }}
  ctx.textBaseline = "top";
  for (const time of niceTicks(state.start, state.end, 7)) {{
    const x = timeToX(canvas, time);
    ctx.beginPath(); ctx.moveTo(x, margins.top); ctx.lineTo(x, bottom); ctx.stroke();
    ctx.textAlign = "center";
    ctx.fillText(formatTime(time), x, bottom + 6);
  }}
  ctx.save();
  ctx.translate(11, (margins.top + bottom) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.fillText(unit, 0, 0);
  ctx.restore();
}}
function drawSeries(ctx, canvas, height, times, values, low, high, color) {{
  const [first, last] = visibleIndices(times);
  const bottom = height - margins.bottom;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.7;
  ctx.beginPath();
  let open = false;
  for (let i = first; i <= last; i++) {{
    const value = values[i];
    if (value === null || !Number.isFinite(value)) {{ open = false; continue; }}
    const x = timeToX(canvas, times[i]);
    const y = margins.top + (high - value) / (high - low) * (bottom - margins.top);
    if (!open) {{ ctx.moveTo(x, y); open = true; }} else ctx.lineTo(x, y);
  }}
  ctx.stroke();
}}
function drawCursor(ctx, canvas, height) {{
  const time = state.hoverTime === null ? audio.currentTime : state.hoverTime;
  if (time < state.start || time > state.end) return;
  const x = timeToX(canvas, time);
  ctx.strokeStyle = colors.cursor;
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(x, margins.top); ctx.lineTo(x, height - margins.bottom); ctx.stroke();
}}
function drawEnvelope(canvas, id, times, minimum, maximum) {{
  const [ctx, width, height] = prepare(canvas);
  const [low, high] = axisBounds(id, -1, 1);
  drawAxes(ctx, canvas, height, low, high, "amp");
  const [first, last] = visibleIndices(times);
  const bottom = height - margins.bottom;
  ctx.fillStyle = colors.shade;
  ctx.strokeStyle = colors.one;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = first; i <= last; i++) {{
    const x = timeToX(canvas, times[i]);
    const y = margins.top + (high - maximum[i]) / (high - low) * (bottom - margins.top);
    if (i === first) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }}
  for (let i = last; i >= first; i--) {{
    const x = timeToX(canvas, times[i]);
    const y = margins.top + (high - minimum[i]) / (high - low) * (bottom - margins.top);
    ctx.lineTo(x, y);
  }}
  ctx.closePath(); ctx.fill(); ctx.stroke();
  drawCursor(ctx, canvas, height);
}}
function spectrumColor(value, low, high) {{
  const ratio = clamp((value - low) / Math.max(high - low, 1), 0, 1);
  const hue = 245 - ratio * 205;
  const light = 16 + ratio * 58;
  return `hsl(${{hue}} 78% ${{light}}%)`;
}}
function drawSpectrum(canvas) {{
  const [ctx, width, height] = prepare(canvas);
  const data = DATA.accompaniment;
  const autoLow = data.frequencies[0];
  const autoHigh = data.frequencies[data.frequencies.length - 1];
  const [low, high] = axisBounds("accompaniment-spectrum", autoLow, autoHigh);
  const plotHeight = height - margins.top - margins.bottom;
  const innerWidth = Math.max(1, Math.round(plotWidth(canvas)));
  const innerHeight = Math.max(1, Math.round(plotHeight));
  const key = [state.start, state.end, low, high, innerWidth, innerHeight].join("|");
  if (!canvas._spectrumCache || canvas._spectrumCache.key !== key) {{
    const image = document.createElement("canvas");
    image.width = innerWidth;
    image.height = innerHeight;
    const imageContext = image.getContext("2d");
    const times = data.spectrumTimes;
    const [first, last] = visibleIndices(times);
    for (let ti = first; ti <= last; ti++) {{
      const nextTime = ti + 1 < times.length ? times[ti + 1] : times[ti] + DATA.timestep;
      const x0 = (times[ti] - state.start) / visibleDuration() * innerWidth;
      const x1 = (nextTime - state.start) / visibleDuration() * innerWidth;
      for (let fi = 0; fi < data.frequencies.length; fi++) {{
        const frequency = data.frequencies[fi];
        if (frequency < low || frequency > high) continue;
        const nextFrequency = fi + 1 < data.frequencies.length ? data.frequencies[fi + 1] : frequency;
        const y0 = (high - nextFrequency) / (high - low) * innerHeight;
        const y1 = (high - frequency) / (high - low) * innerHeight;
        imageContext.fillStyle = spectrumColor(data.spectrum[fi][ti], data.spectrumMin, data.spectrumMax);
        imageContext.fillRect(x0, y0, Math.max(1, x1 - x0 + 0.5), Math.max(1, y1 - y0 + 0.5));
      }}
    }}
    canvas._spectrumCache = {{ key, image }};
  }}
  ctx.drawImage(canvas._spectrumCache.image, margins.left, margins.top, innerWidth, innerHeight);
  drawAxes(ctx, canvas, height, low, high, "Hz");
  drawCursor(ctx, canvas, height);
}}
function drawTrack(canvas, track) {{
  const [ctx, width, height] = prepare(canvas);
  const times = track.times || DATA.times;
  const [first, last] = visibleIndices(times);
  const automatic = boundsFor(track, first, last);
  const [low, high] = axisBounds(track.id, automatic[0], automatic[1]);
  drawAxes(ctx, canvas, height, low, high, track.unit);
  track.series.forEach((series, index) => drawSeries(
    ctx, canvas, height, times, series.values, low, high, index ? colors.two : colors.one
  ));
  drawCursor(ctx, canvas, height);
}}
function drawAll() {{
  for (const canvas of canvases) {{
    const id = canvas.dataset.track;
    if (id === "wave") drawEnvelope(canvas, id, DATA.times, DATA.waveMin, DATA.waveMax);
    else if (id === "accompaniment-wave") drawEnvelope(canvas, id, DATA.accompaniment.times, DATA.accompaniment.waveMin, DATA.accompaniment.waveMax);
    else if (id === "accompaniment-spectrum") drawSpectrum(canvas);
    else drawTrack(canvas, trackFor(id));
  }}
  document.getElementById("metadata").textContent =
    `${{formatTime(DATA.duration)}} · ${{(DATA.timestep * 1000).toFixed(2)}} ms/frame · view ${{formatTime(state.start)}}–${{formatTime(state.end)}}`;
}}
function updateReadouts() {{
  const time = state.hoverTime === null ? audio.currentTime : state.hoverTime;
  const index = indexAt(time);
  const wave = document.querySelector('[data-value="wave"]');
  wave.textContent = formatTime(time);
  if (DATA.accompaniment) {{
    document.querySelector('[data-value="accompaniment-wave"]').textContent = formatTime(time);
  }}
  for (const track of DATA.tracks) {{
    const values = track.series.map(series => {{
      const value = series.values[index];
      return `${{series.label}} ${{value === null ? "unvoiced" : value.toFixed(2) + " " + track.unit}}`;
    }});
    document.querySelector(`[data-value="${{track.id}}"]`).textContent =
      `${{formatTime(time)}} · ${{values.join(" · ")}}`;
  }}
}}
function followPlayback() {{
  if (!document.getElementById("follow").checked || audio.paused || state.dragging) return;
  const time = audio.currentTime;
  const span = visibleDuration();
  if (span < DATA.duration && (time > state.end - span * 0.12 || time < state.start)) {{
    setRange(time - span * 0.18, time + span * 0.82);
  }}
}}
function animationFrame() {{
  if (accompaniment && !audio.paused && Math.abs(accompaniment.currentTime - audio.currentTime) > 0.08) {{
    accompaniment.currentTime = audio.currentTime;
  }}
  followPlayback();
  updateReadouts();
  drawAll();
  if (!audio.paused) requestAnimationFrame(animationFrame);
}}
audio.addEventListener("play", () => {{
  if (accompaniment) {{
    accompaniment.currentTime = audio.currentTime;
    accompaniment.playbackRate = audio.playbackRate;
    accompaniment.play().catch(() => {{}});
  }}
  requestAnimationFrame(animationFrame);
}});
audio.addEventListener("pause", () => {{
  if (accompaniment) accompaniment.pause();
}});
audio.addEventListener("seeking", () => {{
  if (accompaniment) accompaniment.currentTime = audio.currentTime;
}});
audio.addEventListener("ratechange", () => {{
  if (accompaniment) accompaniment.playbackRate = audio.playbackRate;
}});
audio.addEventListener("ended", () => {{
  if (accompaniment) {{
    accompaniment.pause();
    accompaniment.currentTime = 0;
  }}
}});
if (accompaniment && accompanimentVolume) {{
  accompaniment.volume = Number(accompanimentVolume.value);
  accompanimentVolume.addEventListener("input", () => {{
    accompaniment.volume = Number(accompanimentVolume.value);
  }});
}}
audio.addEventListener("timeupdate", () => {{ updateReadouts(); if (audio.paused) drawAll(); }});
document.getElementById("zoom-in").addEventListener("click", () => zoom(0.6, audio.currentTime || (state.start + state.end) / 2));
document.getElementById("zoom-out").addEventListener("click", () => zoom(1.67, audio.currentTime || (state.start + state.end) / 2));
document.getElementById("full-view").addEventListener("click", () => setRange(0, DATA.duration));
document.getElementById("axis-apply").addEventListener("click", () => {{
  const low = Number(axisMin.value);
  const high = Number(axisMax.value);
  if (Number.isFinite(low) && Number.isFinite(high) && high > low) {{
    state.yRanges[axisTrack.value] = [low, high];
    drawAll();
  }}
}});
document.getElementById("axis-auto").addEventListener("click", () => {{
  delete state.yRanges[axisTrack.value];
  axisMin.value = "";
  axisMax.value = "";
  drawAll();
}});
axisTrack.addEventListener("change", () => {{
  const range = state.yRanges[axisTrack.value];
  axisMin.value = range ? range[0] : "";
  axisMax.value = range ? range[1] : "";
  drawAll();
}});
new ResizeObserver(drawAll).observe(root);
updateReadouts();
drawAll();
</script>
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")
    return output_path
