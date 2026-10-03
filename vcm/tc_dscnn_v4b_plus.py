"""TC-DSCNN V4-B+ : upgraded V4-B for the 19-class ai231 intent task.

Changes from V4-B (same log-mel front end and [B,1,64,T] + frame_lengths contract):
  * width 96 -> 128 temporal channels
  * six dilated temporal blocks (1,2,4,8,16,32): ~263-frame (2.6 s) receptive field
  * squeeze-excitation (length-masked) inside every temporal block
  * frequency-collapsed, length-masked attentive statistics pooling (mean + std)
All operations stay padding-invariant, like V2/V4.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .dscnn_exp1 import temporal_mask
from .tc_dscnn_v2 import DSBlock, IntentTCDSCNNV2


def masked_mean(x: Tensor, lengths: Tensor) -> Tensor:
    """Mean over valid (frequency, time) cells of a [B,C,F,T] tensor -> [B,C]."""
    return temporal_mask(x, lengths).sum((2, 3)) / (x.shape[2] * lengths).unsqueeze(1).to(x.dtype)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.reduce = nn.Linear(channels, channels // reduction)
        self.expand = nn.Linear(channels // reduction, channels)

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        scale = torch.sigmoid(self.expand(torch.relu(self.reduce(masked_mean(x, lengths)))))
        return x * scale[:, :, None, None]


class SETemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__(); self.dilation = dilation
        self.depthwise = nn.Conv2d(channels, channels, (1, 3), padding=(0, dilation),
                                   dilation=(1, dilation), groups=channels, bias=False)
        self.dn = nn.BatchNorm2d(channels)
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.pn = nn.BatchNorm2d(channels)
        self.se = SqueezeExcitation(channels)
        self.drop = nn.Dropout2d(dropout); self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        residual = temporal_mask(x, lengths)
        x = temporal_mask(self.activation(self.dn(self.depthwise(residual))), lengths)
        x = temporal_mask(self.pn(self.pointwise(x)), lengths)
        x = self.drop(self.se(x, lengths))
        return temporal_mask(self.activation(x + residual), lengths)


class AttentiveStatsPool(nn.Module):
    """Length-masked attentive statistics pooling over time on [B,C,T]."""
    def __init__(self, channels: int, hidden: int = 64) -> None:
        super().__init__()
        self.attention = nn.Sequential(nn.Conv1d(channels, hidden, 1), nn.Tanh(), nn.Conv1d(hidden, channels, 1))

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        valid = torch.arange(x.shape[-1], device=x.device)[None, None, :] < lengths[:, None, None]
        weights = self.attention(x).float().masked_fill(~valid, -torch.inf).softmax(-1)
        x = x.float()
        mean = (weights * x).sum(-1)
        std = ((weights * x.square()).sum(-1) - mean.square()).clamp_min(1e-5).sqrt()
        return torch.cat((mean, std), dim=1)


class IntentTCDSCNNV4BPlus(nn.Module):
    dilations = (1, 2, 4, 8, 16, 32)

    def __init__(self, num_classes: int = 19, dropout: float = 0.30, width: int = 128,
                 block_dropout: float = 0.05) -> None:
        super().__init__()
        self.stem = nn.Conv2d(1, 64, 3, stride=(2, 2), padding=1, bias=False)
        self.stem_norm = nn.BatchNorm2d(64); self.activation = nn.ReLU(inplace=True)
        self.feature_blocks = nn.ModuleList((DSBlock(64, width, 2), DSBlock(width, width, 2)))
        self.temporal_blocks = nn.ModuleList(SETemporalBlock(width, d, block_dropout) for d in self.dilations)
        self.pool = AttentiveStatsPool(width)
        self.dropout = nn.Dropout(dropout); self.classifier = nn.Linear(2 * width, num_classes)

    output_lengths = staticmethod(IntentTCDSCNNV2.output_lengths)

    def forward_features(self, features: Tensor, frame_lengths: Tensor) -> tuple[Tensor, Tensor]:
        features = temporal_mask(features, frame_lengths); lengths = self.output_lengths(frame_lengths)
        x = temporal_mask(self.activation(self.stem_norm(self.stem(features))), lengths)
        for block in self.feature_blocks: x = block(x, lengths)
        for block in self.temporal_blocks: x = block(x, lengths)
        return x, lengths

    def forward(self, features: Tensor, frame_lengths: Tensor) -> Tensor:
        if features.ndim != 4 or features.shape[1:3] != (1, 64):
            raise ValueError(f"Expected [B,1,64,T], got {tuple(features.shape)}")
        if frame_lengths.shape != (features.shape[0],):
            raise ValueError("Expected one valid frame length per utterance")
        x, lengths = self.forward_features(features, frame_lengths)
        pooled = self.pool(x.mean(2), lengths)  # collapse frequency (8 bins), pool time
        return self.classifier(self.dropout(pooled))


def architecture_config(width: int = 128) -> dict[str, object]:
    return {
        "family": "temporal residual depthwise-separable CNN (V4-B upgrade)", "recurrent": False,
        "stem": "Conv2D(1,64,3x3,stride=2x2,padding=1)-BatchNorm-ReLU",
        "feature_channels": [64, width, width], "temporal_channels": width,
        "temporal_dilations": [1, 2, 4, 8, 16, 32], "temporal_kernel": [1, 3],
        "squeeze_excitation": "per temporal block, reduction 4, length-masked",
        "input_frame_receptive_field": 263,
        "pooling": "frequency mean -> length-masked attentive statistics (mean+std)",
        "classifier": f"Linear({2 * width},19) raw logits", "dropout": 0.30, "block_dropout2d": 0.05,
        "change_from_v4b": "width 96->128, +1 dilated block (32), SE, attentive stats pooling",
    }
