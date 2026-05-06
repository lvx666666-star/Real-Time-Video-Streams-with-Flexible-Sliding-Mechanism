from typing import Dict

from .common import require_torch

torch, nn = require_torch()


class ClassificationHead(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, num_target_classes: int = None) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.num_target_classes = num_classes - 2 if num_target_classes is None else num_target_classes
        self.fc = nn.Linear(feature_dim, num_classes)

    @property
    def label_schema(self) -> Dict[str, int]:
        return {
            "target_start": 0,
            "target_end": self.num_target_classes - 1,
            "background": self.num_target_classes,
            "vague": self.num_target_classes + 1,
        }

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)

