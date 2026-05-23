"""ResNet18 implementation for CIFAR-10 classification."""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Basic ResNet block: two 3x3 conv layers with residual connection.

    Unlike ResNet50's bottleneck design, ResNet18 uses simple 3x3→3x3 blocks
    which are more efficient for smaller networks and CIFAR datasets.
    """

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module = None,
    ) -> None:
        """Initialize BasicBlock.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            stride: Stride for the first conv layer.
            downsample: Optional downsampling module for residual path.
        """
        super().__init__()

        # Main path: two 3x3 conv layers with batch norm
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Shortcut path
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through basic block."""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    """ResNet-18 network architecture for CIFAR-10.

    Architecture overview:
        - Initial stem: 3x3 conv (64 ch)
        - Stage 1 (C1): 2x basic blocks, base channels 64, output 64 ch
        - Stage 2 (C2): 2x basic blocks, base channels 128, output 128 ch
        - Stage 3 (C3): 2x basic blocks, base channels 256, output 256 ch
        - Stage 4 (C4): 2x basic blocks, base channels 512, output 512 ch
        - Classification: global average pool + linear

    Total layers: 1 (stem conv) + 8 (basic blocks * 2 conv) + 1 (fc) = 18 conv layers
    """

    def __init__(
        self,
        num_classes: int = 10,
        block: nn.Module = BasicBlock,
        layers: list = [2, 2, 2, 2],
        zero_init_residual: bool = False,
    ) -> None:
        """Initialize ResNet18.

        Args:
            num_classes: Number of output classes (default 10 for CIFAR-10).
            block: Block type to use (default BasicBlock).
            layers: Number of basic blocks in each stage.
            zero_init_residual: If True, initialize last BN in each block to zero.
        """
        super().__init__()

        # Initial stem (CIFAR-friendly: no max pool to preserve small feature maps)
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Define channels for each stage
        self.layer1_channels = 64  # output = 64 * 1 = 64
        self.layer2_channels = 128  # output = 128 * 1 = 128
        self.layer3_channels = 256  # output = 256 * 1 = 256
        self.layer4_channels = 512  # output = 512 * 1 = 512

        # Stages: stride=2 for downsampling on first block of each stage
        self.layer1 = self._make_layer(block, self.layer1_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(block, self.layer2_channels, layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.layer3_channels, layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.layer4_channels, layers[3], stride=2)

        # Final classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Weight initialization
        self._initialize_weights()

    def _make_layer(
        self,
        block: nn.Module,
        channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """Create a sequence of basic blocks forming one stage.

        Args:
            block: Block type (BasicBlock).
            channels: Base number of channels for this stage.
            num_blocks: Number of blocks in this stage.
            stride: Stride of the first block (controls spatial downsampling).

        Returns:
            nn.Sequential containing the blocks.
        """

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for i, s in enumerate(strides):
            # Create downsample module only for the first block
            if i == 0 and (
                stride != 1 or self.in_channels != channels * block.expansion
            ):
                downsample = nn.Sequential(
                    nn.Conv2d(
                        self.in_channels,
                        channels * block.expansion,
                        1,
                        stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(channels * block.expansion),
                )
            else:
                downsample = None

            layers.append(block(self.in_channels, channels, s, downsample))
            self.in_channels = channels * block.expansion

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet18.

        Args:
            x: Input tensor (B, 3, H, W).

        Returns:
            Classification logits (B, num_classes).
        """
        # Initial stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Classification head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def model_factory(num_classes: int = 10) -> nn.Module:
    """Factory function to create a ResNet18 model instance."""
    return ResNet18(num_classes=num_classes)
