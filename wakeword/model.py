from __future__ import annotations

import torch
from torch import Tensor, nn


class DepthwiseSeparable(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int]) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class MarvinDSCNN(nn.Module):
    """Small depthwise-separable CNN returning one MARVIN logit per clip."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            DepthwiseSeparable(16, 24, stride=(1, 1)),
            DepthwiseSeparable(24, 32, stride=(2, 2)),
            DepthwiseSeparable(32, 32, stride=(1, 1)),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.1),
            nn.Linear(32, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features).squeeze(1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
