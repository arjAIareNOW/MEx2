from __future__ import annotations

from torch import Tensor, nn

from .dscnn_exp1 import conv_output_lengths, temporal_mask


class DSBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, frequency_stride: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, stride=(frequency_stride, 1),
                                   padding=1, groups=in_channels, bias=False)
        self.dn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pn = nn.BatchNorm2d(out_channels); self.activation = nn.ReLU(inplace=True)
    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        x = temporal_mask(self.activation(self.dn(self.depthwise(x))), lengths)
        return temporal_mask(self.activation(self.pn(self.pointwise(x))), lengths)


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__(); self.dilation = dilation
        self.depthwise = nn.Conv2d(channels, channels, (1, 3), padding=(0, dilation),
                                   dilation=(1, dilation), groups=channels, bias=False)
        self.dn = nn.BatchNorm2d(channels)
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.pn = nn.BatchNorm2d(channels); self.activation = nn.ReLU(inplace=True)
    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        residual = temporal_mask(x, lengths)
        x = temporal_mask(self.activation(self.dn(self.depthwise(residual))), lengths)
        x = self.pn(self.pointwise(x))
        return temporal_mask(self.activation(x + residual), lengths)


class IntentTCDSCNNV2(nn.Module):
    dilations = (1, 2, 4, 8, 16)
    def __init__(self, num_classes: int = 20, dropout: float = 0.20) -> None:
        super().__init__()
        self.stem = nn.Conv2d(1, 64, 3, stride=(2, 2), padding=1, bias=False)
        self.stem_norm = nn.BatchNorm2d(64); self.activation = nn.ReLU(inplace=True)
        self.feature_blocks = nn.ModuleList((DSBlock(64, 96, 2), DSBlock(96, 96, 2)))
        self.temporal_blocks = nn.ModuleList(TemporalResidualBlock(96, d) for d in self.dilations)
        self.dropout = nn.Dropout(dropout); self.classifier = nn.Linear(96, num_classes)
    @staticmethod
    def output_lengths(frame_lengths: Tensor) -> Tensor:
        return conv_output_lengths(frame_lengths, 3, 2, 1)
    def forward_features(self, features: Tensor, frame_lengths: Tensor) -> tuple[Tensor, Tensor]:
        features = temporal_mask(features, frame_lengths); lengths = self.output_lengths(frame_lengths)
        x = temporal_mask(self.activation(self.stem_norm(self.stem(features))), lengths)
        for block in self.feature_blocks: x = block(x, lengths)
        for block in self.temporal_blocks: x = block(x, lengths)
        return x, lengths
    def forward(self, features: Tensor, frame_lengths: Tensor) -> Tensor:
        if features.ndim != 4 or features.shape[1:3] != (1, 64):
            raise ValueError(f"Expected [B,1,64,T], got {tuple(features.shape)}")
        x, lengths = self.forward_features(features, frame_lengths); x = temporal_mask(x, lengths)
        pooled = x.sum((2, 3)) / (x.shape[2] * lengths).unsqueeze(1).to(x.dtype)
        return self.classifier(self.dropout(pooled))


def architecture_config() -> dict[str, object]:
    return {
        "family": "temporal residual depthwise-separable CNN", "recurrent": False,
        "stem": "Conv2D(1,64,3x3,stride=2x2,padding=1)-BatchNorm-ReLU",
        "feature_channels": [64, 96, 96], "temporal_channels": 96,
        "temporal_dilations": [1, 2, 4, 8, 16], "temporal_kernel": [1, 3],
        "residual": True, "input_frame_receptive_field": 135,
        "pooling": "length-aware masked global average", "dropout": 0.20,
        "classifier": "Linear(96,20) raw logits", "causal": False, "streaming": False,
    }


if __name__ == "__main__":
    from .v2_experiment_common import ExperimentSpec, main_for_experiment
    main_for_experiment(ExperimentSpec("TC-DSCNN V2", "IntentTCDSCNNV2", "vcm_tc_dscnn_v2",
                                       IntentTCDSCNNV2, architecture_config, "pure convolutional temporal"))
