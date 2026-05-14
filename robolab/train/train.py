"""Training script for ConvNet on CIFAR-10."""

import torch
import torch.nn as nn

from robolab.configs import Hyperparameters
from robolab.data import get_train_loader
from robolab.models import ConvNet
from robolab.utils import get_device

CIFAR10_CLASSES = 10


def train(checkpoint_dir: str = "checkpoints", data_root: str = "./data") -> None:
    """Train the ConvNet model on CIFAR-10.

    Args:
        checkpoint_dir: Directory to save model checkpoints.
        data_root: Root directory for dataset storage.
    """
    hp = Hyperparameters()
    device = get_device()

    # Data
    train_loader = get_train_loader(data_root=data_root)

    # Model
    model = ConvNet(num_classes=CIFAR10_CLASSES).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.learning_rate)

    total_step = len(train_loader)
    for epoch in range(hp.num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, 3, 32, 32).to(device)
            labels = labels.to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{hp.num_epochs}], "
                    f"Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}"
                )

    # Save checkpoint
    torch.save(model.state_dict(), f"{checkpoint_dir}/model.ckpt")
    print(f"Checkpoint saved to {checkpoint_dir}/model.ckpt")


if __name__ == "__main__":
    train()
