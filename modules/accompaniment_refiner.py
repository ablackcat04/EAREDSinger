from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from basics.base_module import CategorizedModule


class LocalResidualBlock(nn.Module):
    """A local temporal block that does not normalize away input loudness."""

    def __init__(self, channels: int, kernel_size: int, dropout: float):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Local encoder kernel_size must be odd.")
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.pointwise(self.depthwise(F.silu(x))).chunk(2, dim=1)
        return x + self.dropout(value * torch.sigmoid(gate))


class LearnedMelDownsampler(nn.Module):
    """Encode full-resolution mel frames before reducing their time resolution."""

    def __init__(
        self,
        mel_bins: int,
        hidden_size: int,
        downsample_factor: int,
        num_local_layers: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        if downsample_factor <= 0 or downsample_factor & (downsample_factor - 1):
            raise ValueError("downsample_factor must be a positive power of two.")
        if num_local_layers < 0:
            raise ValueError("num_local_layers must not be negative.")

        self.downsample_factor = downsample_factor
        self.mel_proj = nn.Conv1d(
            mel_bins, hidden_size, kernel_size, padding=kernel_size // 2
        )
        self.level_proj = nn.Conv1d(1, hidden_size, 1)
        self.input_blocks = nn.ModuleList(
            LocalResidualBlock(hidden_size, kernel_size, dropout)
            for _ in range(num_local_layers)
        )
        self.downsample_stages = nn.ModuleList()
        for _ in range(int(math.log2(downsample_factor))):
            self.downsample_stages.append(
                nn.ModuleDict(
                    {
                        "conv": nn.Conv1d(
                            hidden_size, hidden_size, 4, stride=2, padding=1
                        ),
                        "block": LocalResidualBlock(
                            hidden_size, kernel_size, dropout
                        ),
                    }
                )
            )

    def forward(self, mel: torch.Tensor, level: torch.Tensor) -> torch.Tensor:
        x = self.mel_proj(mel.transpose(1, 2))
        x = x + self.level_proj(level.transpose(1, 2))
        for block in self.input_blocks:
            x = block(x)
        for stage in self.downsample_stages:
            x = F.silu(stage["conv"](x))
            x = stage["block"](x)
        return x.transpose(1, 2)


class AccompanimentTensionRefiner(CategorizedModule):
    """Predict a whole-song tension residual from a full-resolution accompaniment mel."""

    @property
    def category(self):
        return "tension_refiner"

    def __init__(
        self,
        *,
        mel_bins: int,
        hidden_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_size: int = 1024,
        downsample_factor: int = 16,
        num_local_layers: int = 2,
        local_kernel_size: int = 5,
        dropout: float = 0.1,
        initial_gate: float = 0.1,
        mel_timestep: float = 512 / 44100,
        time_scale_seconds: float = 300.0,
        mel_mean: Sequence[float] | torch.Tensor | None = None,
        mel_std: Sequence[float] | torch.Tensor | None = None,
        mel_level_mean: float = 0.0,
        mel_level_std: float = 1.0,
        tension_min: float = -10.0,
        tension_max: float = 10.0,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")
        if num_layers <= 0 or num_heads <= 0 or ffn_size <= 0:
            raise ValueError(
                "Transformer layer, head, and feed-forward sizes must be positive."
            )
        if not 0.0 < initial_gate < 1.0:
            raise ValueError("initial_gate must be between 0 and 1.")
        if mel_timestep <= 0 or time_scale_seconds <= 0:
            raise ValueError("Mel timestep and time scale must be positive.")

        mean = (
            torch.zeros(mel_bins, dtype=torch.float32)
            if mel_mean is None
            else torch.as_tensor(mel_mean, dtype=torch.float32)
        )
        std = (
            torch.ones(mel_bins, dtype=torch.float32)
            if mel_std is None
            else torch.as_tensor(mel_std, dtype=torch.float32)
        )
        if mean.shape != (mel_bins,) or std.shape != (mel_bins,):
            raise ValueError(
                f"Expected mel normalization statistics with shape ({mel_bins},)."
            )
        if (
            not torch.isfinite(mean).all()
            or not torch.isfinite(std).all()
            or (std <= 0).any()
        ):
            raise ValueError(
                "Mel normalization statistics must be finite and std must be positive."
            )
        if (
            not math.isfinite(mel_level_mean)
            or not math.isfinite(mel_level_std)
            or mel_level_std <= 0
        ):
            raise ValueError(
                "Mel level normalization statistics must be finite with positive std."
            )

        self.downsample_factor = downsample_factor
        self.mel_timestep = mel_timestep
        self.time_scale_seconds = time_scale_seconds
        self.tension_min = tension_min
        self.tension_max = tension_max
        # Fixed training-set statistics preserve broadband amplitude changes.
        self.register_buffer("mel_mean", mean)
        self.register_buffer("mel_std", std)
        self.register_buffer(
            "mel_level_mean", torch.tensor(float(mel_level_mean))
        )
        self.register_buffer("mel_level_std", torch.tensor(float(mel_level_std)))

        self.mel_encoder = LearnedMelDownsampler(
            mel_bins=mel_bins,
            hidden_size=hidden_size,
            downsample_factor=downsample_factor,
            num_local_layers=num_local_layers,
            kernel_size=local_kernel_size,
            dropout=dropout,
        )
        self.base_proj = nn.Linear(1, hidden_size)
        # Relative position, phase, elapsed time, duration, and curve coverage.
        self.time_proj = nn.Linear(6, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ffn_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_size),
            enable_nested_tensor=False,
        )
        self.out_proj = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.gate_logit = nn.Parameter(
            torch.tensor(
                math.log(initial_gate / (1.0 - initial_gate)), dtype=torch.float32
            )
        )

    def _validate_lengths(
        self,
        mel: torch.Tensor,
        base_tension: torch.Tensor,
        mel_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        if mel.ndim != 3 or base_tension.ndim != 2:
            raise ValueError("Expected mel [B, T, M] and base tension [B, T].")
        if mel.shape[:2] != base_tension.shape:
            raise ValueError("Mel and base tension batch/time dimensions must match.")
        if mel.shape[1] == 0:
            raise ValueError("Mel and tension sequences must not be empty.")
        if mel_lengths is None:
            mel_lengths = torch.full(
                (mel.shape[0],),
                mel.shape[1],
                dtype=torch.long,
                device=mel.device,
            )
        else:
            mel_lengths = mel_lengths.to(device=mel.device, dtype=torch.long)
        if mel_lengths.shape != (mel.shape[0],):
            raise ValueError("mel_lengths must have shape [B].")
        if (mel_lengths <= 0).any() or (mel_lengths > mel.shape[1]).any():
            raise ValueError(
                "Every mel length must be within the padded time dimension."
            )
        return mel_lengths

    def forward(
        self,
        mel_accompaniment: torch.Tensor,
        base_tension: torch.Tensor,
        curve_mask: torch.Tensor | None = None,
        mel_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            mel_accompaniment: Full-resolution full-song mel, ``[B, T, M]``.
            base_tension: Frozen base-model curve, ``[B, T]``.
            curve_mask: Frames where a vocal tension curve exists, ``[B, T]``.
            mel_lengths: Unpadded full-song mel lengths, ``[B]``.

        Returns:
            ``(refined_tension, full_resolution_delta, coarse_delta)``.
        """
        mel_lengths = self._validate_lengths(
            mel_accompaniment, base_tension, mel_lengths
        )
        batch_size, full_length, mel_bins = mel_accompaniment.shape
        if mel_bins != self.mel_mean.numel():
            raise ValueError(
                f"Expected {self.mel_mean.numel()} mel bins, received {mel_bins}."
            )
        if curve_mask is None:
            curve_mask = torch.ones_like(base_tension, dtype=torch.bool)
        elif curve_mask.shape != base_tension.shape:
            raise ValueError("curve_mask and base_tension shapes do not match.")
        curve_mask = curve_mask.bool()

        frame_index = torch.arange(
            full_length, device=mel_accompaniment.device
        )[None]
        valid_frames = frame_index < mel_lengths[:, None]
        normalized_mel = (mel_accompaniment - self.mel_mean) / self.mel_std
        # Mel values are natural-log magnitudes. This explicit log-mean-exp
        # feature exposes broadband amplitude before hidden-state LayerNorms.
        frame_level = torch.logsumexp(
            mel_accompaniment, dim=-1, keepdim=True
        ) - math.log(mel_bins)
        normalized_level = (
            frame_level - self.mel_level_mean
        ) / self.mel_level_std
        normalized_mel = normalized_mel.masked_fill(
            ~valid_frames[..., None], 0.0
        )
        normalized_level = normalized_level.masked_fill(
            ~valid_frames[..., None], 0.0
        )

        padded_length = (
            math.ceil(full_length / self.downsample_factor)
            * self.downsample_factor
        )
        pad_length = padded_length - full_length
        if pad_length:
            normalized_mel = F.pad(
                normalized_mel.transpose(1, 2), (0, pad_length)
            ).transpose(1, 2)
            normalized_level = F.pad(
                normalized_level.transpose(1, 2), (0, pad_length)
            ).transpose(1, 2)
            base_padded = F.pad(base_tension, (0, pad_length))
            curve_padded = F.pad(curve_mask, (0, pad_length))
        else:
            base_padded = base_tension
            curve_padded = curve_mask

        x = self.mel_encoder(normalized_mel, normalized_level)
        coarse_length = x.shape[1]
        expected_coarse_length = padded_length // self.downsample_factor
        if coarse_length != expected_coarse_length:
            raise RuntimeError(
                f"Mel encoder produced {coarse_length} tokens; "
                f"expected {expected_coarse_length}."
            )

        base_blocks = base_padded.reshape(
            batch_size, coarse_length, self.downsample_factor
        )
        curve_blocks = curve_padded.reshape(
            batch_size, coarse_length, self.downsample_factor
        )
        coverage = curve_blocks.to(x.dtype).mean(dim=-1, keepdim=True)
        curve_count = curve_blocks.sum(dim=-1, keepdim=True).clamp_min(1)
        base_coarse = (
            (base_blocks * curve_blocks).sum(dim=-1, keepdim=True) / curve_count
        )

        coarse_lengths = torch.div(
            mel_lengths + self.downsample_factor - 1,
            self.downsample_factor,
            rounding_mode="floor",
        )
        coarse_index = torch.arange(coarse_length, device=x.device)[None].expand(
            batch_size, -1
        )
        coarse_padding_mask = coarse_index >= coarse_lengths[:, None]
        relative_position = coarse_index / (
            coarse_lengths[:, None] - 1
        ).clamp_min(1)
        relative_position = relative_position.to(x.dtype).unsqueeze(-1)
        phase = relative_position * (2.0 * math.pi)
        elapsed = (
            coarse_index.to(x.dtype)
            * self.downsample_factor
            * self.mel_timestep
            / self.time_scale_seconds
        ).unsqueeze(-1)
        duration = (
            (mel_lengths - 1).clamp_min(0).to(x.dtype)
            * self.mel_timestep
            / self.time_scale_seconds
        )[:, None, None].expand(-1, coarse_length, -1)
        time_features = torch.cat(
            [
                relative_position,
                torch.sin(phase),
                torch.cos(phase),
                elapsed,
                duration,
                coverage,
            ],
            dim=-1,
        )

        x = (
            x
            + self.base_proj(base_coarse.to(x.dtype))
            + self.time_proj(time_features)
        )
        x = x.masked_fill(coarse_padding_mask[..., None], 0.0)
        # No causal mask: every token sees the entire song, including the future.
        x = self.transformer(x, src_key_padding_mask=coarse_padding_mask)
        coarse_delta = self.out_proj(x).squeeze(-1) * torch.sigmoid(
            self.gate_logit
        )
        coarse_delta = coarse_delta.masked_fill(coarse_padding_mask, 0.0)
        full_delta = F.interpolate(
            coarse_delta[:, None],
            size=padded_length,
            mode="linear",
            align_corners=False,
        ).squeeze(1)[:, :full_length]
        full_delta = (
            full_delta
            * curve_mask.to(full_delta.dtype)
            * valid_frames.to(full_delta.dtype)
        )
        refined = torch.clamp(
            base_tension + full_delta,
            min=self.tension_min,
            max=self.tension_max,
        )
        return refined, full_delta, coarse_delta
