import torch
import torch.nn as nn


class ConvNet(nn.Module):
    """A lightweight convolutional neural network for image classification.

    This model is designed for CIFAR-10 classification tasks. It consists
    of two convolutional blocks followed by fully connected layers. Each
    convolutional block includes batch normalization, activation functions,
    and max pooling for regularization and feature extraction.

    Args:
        num_classes (int): Number of output classes for classification.

    Example:
        >>> model = ConvNet(num_classes=10)
        >>> x = torch.randn(1, 3, 32, 32)
        >>> output = model(x)
    """

    def __init__(self, num_classes: int) -> None:
        """Initialize the ConvNet model.

        Constructs the network architecture with two convolutional blocks,
        a dropout layer, and two fully connected layers.

        Args:
            num_classes (int): Number of output classes for classification.
        """
        super(ConvNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.GELU(),
            nn.MaxPool2d(2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(8 * 8 * 64, 8 * 64)  # 8x smaller than the original size
        self.fc2 = nn.Linear(8 * 64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the network.

        Processes the input tensor through two convolutional blocks,
        flattens the output, applies dropout, and passes through
        fully connected layers.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).
                Expected shape for CIFAR-10: (N, 3, 32, 32).

        Returns:
            torch.Tensor: Output logits tensor of shape (N, num_classes).
        """

        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
