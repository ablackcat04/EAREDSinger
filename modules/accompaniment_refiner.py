from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from basics.base_module import CategorizedModule


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.norm = nn.GroupNorm(1, channels)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(self.norm(x))
        value, gate = self.pointwise(x).chunk(2, dim=1)
        x = value * torch.sigmoid(gate)
        return residual + self.dropout(x)


class AccompanimentTensionRefiner(CategorizedModule):
    """Predict a coarse, interpretable tension residual from a full accompaniment."""

    @property
    def category(self):
        return "tension_refiner"

    def __init__(
        self,
        *,
        mel_bins: int,
        hidden_size: int = 256,
        num_layers: int = 8,
        kernel_size: int = 5,
        dropout: float = 0.1,
        initial_gate: float = 0.1,
        tension_min: float = -10.0,
        tension_max: float = 10.0,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd so temporal length is preserved.")
        if not 0.0 < initial_gate < 1.0:
            raise ValueError("initial_gate must be between 0 and 1.")

        self.tension_min = tension_min
        self.tension_max = tension_max
        self.mel_norm = nn.LayerNorm(mel_bins)
        self.mel_proj = nn.Linear(mel_bins, hidden_size)
        self.base_proj = nn.Linear(1, hidden_size)
        # normalized position, sin(position), cos(position), and vocal-curve coverage
        self.time_proj = nn.Linear(4, hidden_size)
        self.blocks = nn.ModuleList(
            ResidualTemporalBlock(
                hidden_size,
                kernel_size,
                dilation=2 ** (layer_index % 4),
                dropout=dropout,
            )
            for layer_index in range(num_layers)
        )
        self.out_norm = nn.GroupNorm(1, hidden_size)
        self.out_proj = nn.Conv1d(hidden_size, 1, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.gate_logit = nn.Parameter(
            torch.tensor(math.log(initial_gate / (1.0 - initial_gate)), dtype=torch.float32)
        )

    def forward(
        self,
        mel_accompaniment: torch.Tensor,
        base_tension: torch.Tensor,
        curve_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            mel_accompaniment: Downsampled full-song mel, ``[B, T_coarse, M]``.
            base_tension: Frozen base-model curve, ``[B, T]``.
            curve_mask: Frames where a vocal tension curve exists, ``[B, T]``.

        Returns:
            ``(refined_tension, full_resolution_delta, coarse_delta)``.
        """
        if mel_accompaniment.ndim != 3 or base_tension.ndim != 2:
            raise ValueError("Expected mel [B, T_coarse, M] and base tension [B, T].")
        if mel_accompaniment.shape[0] != base_tension.shape[0]:
            raise ValueError("Mel and base tension batch sizes do not match.")

        batch_size, coarse_length, _ = mel_accompaniment.shape
        full_length = base_tension.shape[1]
        if coarse_length == 0 or full_length == 0:
            raise ValueError("Mel and tension sequences must not be empty.")
        if curve_mask is None:
            curve_mask = torch.ones_like(base_tension, dtype=torch.bool)
        elif curve_mask.shape != base_tension.shape:
            raise ValueError("curve_mask and base_tension shapes do not match.")

        base_coarse = F.adaptive_avg_pool1d(base_tension[:, None], coarse_length).transpose(1, 2)
        coverage = F.adaptive_avg_pool1d(
            curve_mask.float()[:, None], coarse_length
        ).transpose(1, 2)
        position = torch.linspace(
            0.0,
            1.0,
            coarse_length,
            dtype=mel_accompaniment.dtype,
            device=mel_accompaniment.device,
        )[None, :, None].expand(batch_size, -1, -1)
        phase = position * (2.0 * math.pi)
        time_features = torch.cat(
            [position, torch.sin(phase), torch.cos(phase), coverage], dim=-1
        )

        x = self.mel_proj(self.mel_norm(mel_accompaniment))
        x = x + self.base_proj(base_coarse) + self.time_proj(time_features)
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        coarse_delta = self.out_proj(F.silu(self.out_norm(x))).squeeze(1)
        coarse_delta = coarse_delta * torch.sigmoid(self.gate_logit)
        full_delta = F.interpolate(
            coarse_delta[:, None], size=full_length, mode="linear", align_corners=False
        ).squeeze(1)
        full_delta = full_delta * curve_mask.to(full_delta.dtype)
        refined = torch.clamp(
            base_tension + full_delta,
            min=self.tension_min,
            max=self.tension_max,
        )
        return refined, full_delta, coarse_delta
