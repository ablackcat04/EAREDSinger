from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import torch

from utils.binarizer_utils import (
    SinusoidalSmoothingConv1d,
    get_breathiness,
    get_energy_librosa,
    get_tension_base_harmonic,
    get_voicing,
)
from utils.decomposed_waveform import DecomposedWaveform
from utils.hparams import hparams
from utils.pitch_utils import interp_f0
from utils.binarizer_utils import get_pitch_parselmouth


@dataclass(frozen=True)
class ExtractionConfig:
    sample_rate: int = 44100
    hop_size: int = 512
    win_size: int = 2048
    fft_size: int = 2048
    f0_min: float = 65.0
    f0_max: float = 1100.0
    smooth_width: float = 0.12
    separator: str = "world"
    vr_checkpoint: Path | None = None
    device: str = "cpu"

    @property
    def timestep(self) -> float:
        return self.hop_size / self.sample_rate


@dataclass(frozen=True)
class VarianceFeatures:
    source_path: Path
    sample_rate: int
    hop_size: int
    waveform: np.ndarray
    times: np.ndarray
    f0: np.ndarray
    unvoiced: np.ndarray
    pitch_midi: np.ndarray
    energy: np.ndarray
    breathiness: np.ndarray
    voicing: np.ndarray
    tension: np.ndarray

    @property
    def duration(self) -> float:
        return len(self.waveform) / self.sample_rate


def _smooth(values: np.ndarray, width_seconds: float, timestep: float, device: str) -> np.ndarray:
    kernel_size = max(1, round(width_seconds / timestep))
    if kernel_size == 1:
        return values.astype(np.float32, copy=False)
    smoother = SinusoidalSmoothingConv1d(kernel_size).eval().to(device)
    with torch.no_grad():
        tensor = torch.from_numpy(values.astype(np.float32, copy=False)).to(device)[None]
        return smoother(tensor)[0].cpu().numpy()


def _waveform_frame_count(waveform: np.ndarray, hop_size: int) -> int:
    return max(1, (len(waveform) + hop_size - 1) // hop_size)


def extract_variances(
    audio_path: str | Path,
    config: ExtractionConfig | None = None,
) -> VarianceFeatures:
    """Extract the same frame-level variance features used by DiffSinger binarization."""
    config = config or ExtractionConfig()
    source_path = Path(audio_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {source_path}")
    if config.separator not in {"world", "vr"}:
        raise ValueError("separator must be either 'world' or 'vr'")
    if config.separator == "vr":
        if config.vr_checkpoint is None:
            raise ValueError("The VR separator requires --vr-checkpoint.")
        checkpoint = Path(config.vr_checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"VR checkpoint does not exist: {checkpoint}")
        hparams["hnsep_ckpt"] = str(checkpoint)

    waveform, _ = librosa.load(
        source_path,
        sr=config.sample_rate,
        mono=True,
        dtype=np.float32,
    )
    if waveform.size == 0:
        raise ValueError(f"Audio file is empty: {source_path}")

    length = _waveform_frame_count(waveform, config.hop_size)
    f0_raw, unvoiced = get_pitch_parselmouth(
        waveform,
        samplerate=config.sample_rate,
        length=length,
        hop_size=config.hop_size,
        f0_min=config.f0_min,
        f0_max=config.f0_max,
        interp_uv=False,
    )
    if unvoiced.all():
        raise ValueError(
            "No voiced pitch was detected. Check that the file contains a vocal "
            "performance or adjust --f0-min/--f0-max."
        )

    # DiffSinger interpolates F0 for analysis, while retaining the original UV mask.
    f0, unvoiced = interp_f0(f0_raw, unvoiced)
    pitch_midi = librosa.hz_to_midi(f0.astype(np.float32)).astype(np.float32)

    energy = get_energy_librosa(
        waveform,
        length,
        hop_size=config.hop_size,
        win_size=config.win_size,
    ).astype(np.float32)
    energy = _smooth(energy, config.smooth_width, config.timestep, config.device)

    decomposed = DecomposedWaveform(
        waveform,
        samplerate=config.sample_rate,
        f0=f0 * ~unvoiced,
        hop_size=config.hop_size,
        fft_size=config.fft_size,
        win_size=config.win_size,
        algorithm=config.separator,
        device=config.device,
    )

    breathiness = get_breathiness(
        decomposed, None, None, length=length
    ).astype(np.float32)
    breathiness = _smooth(
        breathiness, config.smooth_width, config.timestep, config.device
    )

    voicing = get_voicing(
        decomposed, None, None, length=length
    ).astype(np.float32)
    voicing = _smooth(
        voicing, config.smooth_width, config.timestep, config.device
    )

    tension = get_tension_base_harmonic(
        decomposed, None, None, length=length, domain="logit"
    ).astype(np.float32)
    tension = _smooth(
        tension, config.smooth_width, config.timestep, config.device
    )

    times = np.arange(length, dtype=np.float64) * config.timestep
    return VarianceFeatures(
        source_path=source_path,
        sample_rate=config.sample_rate,
        hop_size=config.hop_size,
        waveform=waveform,
        times=times,
        f0=f0,
        unvoiced=unvoiced,
        pitch_midi=pitch_midi,
        energy=energy,
        breathiness=breathiness,
        voicing=voicing,
        tension=tension,
    )
