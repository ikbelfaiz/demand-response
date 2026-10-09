from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm
import torch.nn.functional as F


APPLIANCES = ("ac", "water_heater", "washing_machine")


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1,
                 use_weight_norm: bool = True):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=0)
        self.conv = weight_norm(conv) if use_weight_norm else conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_padding, 0)))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(channels, channels, kernel_size, dilation, True),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(channels, channels, kernel_size, dilation, True),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ApplianceHead(nn.Module):
    def __init__(self, in_features: int, hidden: int = 32):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(in_features, hidden), nn.GELU())
        self.power = nn.Linear(hidden, 1)
        self.activity = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        return F.softplus(self.power(h)).squeeze(-1), self.activity(h).squeeze(-1)


class MultiHeadMixin:
    power_scales_w: torch.Tensor

    def _apply_heads(self, encoded: torch.Tensor) -> dict[str, torch.Tensor]:
        scaled, logits = [], []
        for name in APPLIANCES:
            p, a = self.heads[name](encoded)
            scaled.append(p)
            logits.append(a)
        power_scaled = torch.stack(scaled, dim=-1)
        activity_logits = torch.stack(logits, dim=-1)
        scales = self.power_scales_w.to(power_scaled)
        return {
            "power_scaled": power_scaled,
            "power_w": power_scaled * scales,
            "activity_logits": activity_logits,
            "activity_probability": torch.sigmoid(activity_logits),
        }


class CausalMultiTaskTCN(nn.Module, MultiHeadMixin):
    """Six-block causal multi-task TCN with a 253-sample receptive field."""

    receptive_field = 253

    def __init__(self, power_scales_w: list[float] | tuple[float, ...], channels: int = 64,
                 dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32), dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Conv1d(2, channels, 1)
        self.blocks = nn.ModuleList([ResidualBlock(channels, d, 3, dropout) for d in dilations])
        self.heads = nn.ModuleDict({name: ApplianceHead(channels, 32) for name in APPLIANCES})
        self.register_buffer("power_scales_w", torch.tensor(power_scales_w, dtype=torch.float32))

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        z = self.projection(x)
        for block in self.blocks:
            z = block(z)
        return z

    def forward_sequence(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encode_sequence(x).transpose(1, 2)
        return self._apply_heads(encoded)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encode_sequence(x)[:, :, -1]
        return self._apply_heads(encoded)


class CausalCNNBaseline(nn.Module, MultiHeadMixin):
    """Small causal CNN baseline using the same inputs and independent heads."""

    receptive_field = 13

    def __init__(self, power_scales_w: list[float] | tuple[float, ...], channels: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(2, channels, 1), nn.GELU(),
            CausalConv1d(channels, channels, 5, 1, False), nn.GELU(), nn.Dropout(dropout),
            CausalConv1d(channels, channels, 5, 2, False), nn.GELU(), nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict({name: ApplianceHead(channels, 32) for name in APPLIANCES})
        self.register_buffer("power_scales_w", torch.tensor(power_scales_w, dtype=torch.float32))

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward_sequence(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self._apply_heads(self.encode_sequence(x).transpose(1, 2))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self._apply_heads(self.encode_sequence(x)[:, :, -1])


def build_model(kind: str, transforms: dict, model_cfg: dict) -> nn.Module:
    scales = transforms["appliance_power_scales_w"]
    if kind == "tcn":
        return CausalMultiTaskTCN(scales, channels=model_cfg.get("channels", 64),
                                  dilations=tuple(model_cfg.get("dilations", [1, 2, 4, 8, 16, 32])),
                                  dropout=model_cfg.get("dropout", 0.1))
    if kind == "cnn_baseline":
        return CausalCNNBaseline(scales, channels=model_cfg.get("baseline_channels", 32),
                                 dropout=model_cfg.get("dropout", 0.1))
    raise ValueError(f"Unknown model kind: {kind}")
