from __future__ import annotations

import torch
from torch import Tensor, nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


class IntentCNN(nn.Module):
    """Small variable-duration CNN for command/intent classification."""

    def __init__(self, num_classes: int = 20, dropout: float = 0.2) -> None:
        super().__init__()
        self.features = nn.ModuleList(
            (
                ConvBlock(1, 32),
                ConvBlock(32, 64),
                ConvBlock(64, 128),
            )
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, features: Tensor, frame_lengths: Tensor | None = None) -> Tensor:
        x = features
        lengths = frame_lengths
        for block in self.features:
            x = block(x)
            if lengths is not None:
                lengths = torch.div(lengths, 2, rounding_mode="floor").clamp_min(1)

        if lengths is None:
            pooled = self.adaptive_pool(x).flatten(1)
        else:
            time_steps = torch.arange(x.shape[-1], device=x.device).unsqueeze(0)
            mask = (time_steps < lengths.unsqueeze(1)).unsqueeze(1).unsqueeze(2)
            pooled = (x * mask).sum(dim=(2, 3))
            pooled = pooled / (x.shape[2] * lengths).unsqueeze(1).to(x.dtype)
        return self.classifier(self.dropout(pooled))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

