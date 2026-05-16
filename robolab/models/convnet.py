"""ConvNet model definition for CIFAR-10 classification."""

import torch
import torch.nn as nn


class ConvNet(nn.Module):
    """Improved convolutional neural network for CIFAR-10 classification.

    Architecture:
        - Four convolutional blocks with BatchNorm:
          (Conv → BatchNorm → Activation → MaxPool)
        - Global average pooling alternative via flattening
        - Dropout regularization
        - Fully connected classifier

    Example:
        >>> model = ConvNet(num_classes=10)
        >>> x = torch.randn(1, 3, 32, 32)
        >>> output = model(x)  # shape: (1, 10)
    """

    def __init__(self, num_classes: int = 10) -> None:
        """Initialize ConvNet.

        Args:
            num_classes: Number of output classification classes.
        """
        super().__init__()

        self._features = nn.Sequential(
            # Block 1: 3x32x32 -> 32x32x32 -> 32x16x16
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            # Block 2: 32x16x16 -> 64x16x16 -> 64x8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            # Block 3: 64x8x8 -> 128x8x8 -> 128x4x4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
        )

        self._classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (N, 3, 32, 32).

        Returns:
            Logits tensor of shape (N, num_classes).
        """
        x = self._features(x)
        x = x.reshape(x.size(0), -1)
        return self._classifier(x)
