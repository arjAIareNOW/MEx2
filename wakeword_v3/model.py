import torch
from torch import nn
from wakeword.model import DepthwiseSeparable


class MarvinV3(nn.Module):
    """Windowed, non-causal DSCNN; one binary logit, not a general ASR model."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24), nn.ReLU(),
            DepthwiseSeparable(24, 32, (1, 1)),
            DepthwiseSeparable(32, 48, (2, 2)),
            DepthwiseSeparable(48, 48, (1, 1)),
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(96, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.backbone(features)
        representation = torch.cat((x.mean(dim=(2, 3)), x.amax(dim=(2, 3))), dim=1)
        return self.classifier(self.dropout(representation)).squeeze(1)
