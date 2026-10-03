"""Architecture-only V4 experiments; authoritative V2 remains unchanged."""
from __future__ import annotations

from torch import Tensor, nn

from .dscnn_exp1 import temporal_mask
from .tc_dscnn_v2 import IntentTCDSCNNV2, architecture_config as v2_config


class IntentTCDSCNNV4A(IntentTCDSCNNV2):
    """Two additional identity skips around pairs of existing temporal blocks."""
    def forward_features(self, features: Tensor, frame_lengths: Tensor) -> tuple[Tensor, Tensor]:
        features = temporal_mask(features, frame_lengths)
        lengths = self.output_lengths(frame_lengths)
        x = temporal_mask(self.activation(self.stem_norm(self.stem(features))), lengths)
        for block in self.feature_blocks:
            x = block(x, lengths)
        for start in (0, 2):
            residual = x
            x = self.temporal_blocks[start](x, lengths)
            x = self.temporal_blocks[start + 1](x, lengths)
            x = temporal_mask(x + residual, lengths)
        x = self.temporal_blocks[4](x, lengths)
        return x, lengths


class IntentTCDSCNNV4B(IntentTCDSCNNV2):
    """Only final temporal dilation changes: 16 -> 32 (padding 16 -> 32)."""
    dilations = (1, 2, 4, 8, 32)


def architecture_config(variant: str) -> dict[str, object]:
    config = v2_config()
    if variant == "A":
        return config | {
            "change_from_v2": "identity skips around temporal blocks [0,1] and [2,3]",
            "additional_residual_pairs": [[0, 1], [2, 3]],
            "projection": False, "added_parameters": 0,
            "note": "V2 already has per-block residuals and dilation; both are retained unchanged",
        }
    if variant == "B":
        return config | {
            "change_from_v2": "final temporal depthwise dilation/padding 16 -> 32 only",
            "temporal_dilations": [1, 2, 4, 8, 32],
            "input_frame_receptive_field": 199,
            "added_parameters": 0,
            "note": "Existing V2 residuals retained; all temporal output lengths unchanged",
        }
    raise ValueError(variant)


if __name__ == "__main__":
    from .v4_training import main
    main()
