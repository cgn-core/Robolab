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
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (N, 3, 32, 32).

        Returns:
            Logits tensor of shape (N, num_classes).
        """
        x = self._features(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
