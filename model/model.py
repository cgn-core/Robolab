import torch.nn as nn


class ConvNet2(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet2, self).__init__()

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

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
