from .common import require_torch
from .recognizer import ActionRecognizer

torch, nn = require_torch()


class C3DEncoder(nn.Module):
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.output_dim = 512
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)


class C3DRecognizer(ActionRecognizer):
    def __init__(self, num_classes: int, num_target_classes: int = None, in_channels: int = 3) -> None:
        encoder = C3DEncoder(in_channels=in_channels)
        super().__init__(
            encoder=encoder,
            feature_dim=encoder.output_dim,
            num_classes=num_classes,
            num_target_classes=num_target_classes,
        )


C3DClassifier = C3DRecognizer
