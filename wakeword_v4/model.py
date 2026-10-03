"""MARVIN V4 model: residual depthwise-separable CNN with squeeze-excitation.

Same input as V2/V3: [B,1,40,97] V2 log-mel window; one raw MARVIN logit.
"""
import torch
from torch import nn


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(channels, channels // reduction, 1)
        self.expand = nn.Conv2d(channels // reduction, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.mean(dim=(2, 3), keepdim=True)
        return x * torch.sigmoid(self.expand(torch.relu(self.reduce(scale))))


class ResidualDSBlock(nn.Module):
    """depthwise 3x3 -> BN/ReLU -> pointwise -> BN -> SE (+ projected skip) -> ReLU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, dilation: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=dilation,
                                   dilation=dilation, groups=in_channels, bias=False)
        self.dn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pn = nn.BatchNorm2d(out_channels)
        self.se = SqueezeExcitation(out_channels)
        self.skip = nn.Identity() if in_channels == out_channels and stride == 1 else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.dn(self.depthwise(x)))
        y = self.se(self.pn(self.pointwise(y)))
        return torch.relu(y + self.skip(x))


class MarvinV4(nn.Module):
    """Windowed, non-causal residual DSCNN + SE; one binary logit."""

    def __init__(self, width: int = 48, dropout: float = 0.2) -> None:
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv2d(1, w, 3, stride=2, padding=1, bias=False),
                                  nn.BatchNorm2d(w), nn.ReLU())
        self.blocks = nn.Sequential(
            ResidualDSBlock(w, w, 1),
            ResidualDSBlock(w, 2 * w, 2),
            ResidualDSBlock(2 * w, 2 * w, 1),
            ResidualDSBlock(2 * w, 2 * w, 1, dilation=2),
            ResidualDSBlock(2 * w, 2 * w, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(4 * w, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.blocks(self.stem(features))
        representation = torch.cat((x.mean(dim=(2, 3)), x.amax(dim=(2, 3))), dim=1)
        return self.classifier(self.dropout(representation)).squeeze(1)
