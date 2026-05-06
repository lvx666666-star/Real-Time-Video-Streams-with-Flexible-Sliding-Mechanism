from typing import Tuple

from .common import require_torch
from .head import ClassificationHead

torch, nn = require_torch()


class ActionRecognizer(nn.Module):
    def __init__(self, encoder: nn.Module, feature_dim: int, num_classes: int, num_target_classes: int = None) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = ClassificationHead(
            feature_dim=feature_dim,
            num_classes=num_classes,
            num_target_classes=num_target_classes,
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward_logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_logits_from_features(self.forward_features(x))

    @torch.no_grad()
    def infer(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs

