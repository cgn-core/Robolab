"""Training script for ConvNet on CIFAR-10."""

import torch
import torch.nn as nn

from robolab.configs import Hyperparameters
from robolab.data import get_train_loader
from robolab.models import ConvNet
from robolab.utils import get_device, logger


def train(checkpoint_dir: str = "checkpoints", data_root: str = "./data") -> None:
    """Train the ConvNet model on CIFAR-10.

    Args:
        checkpoint_dir: Directory to save model checkpoints.
        data_root: Root directory for dataset storage.
    """
    cfg = Hyperparameters()
    device = get_device()

    # Set random seed for reproducibility
    if cfg.random_seed is not None:
        torch.manual_seed(cfg.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.random_seed)

    # Data
    train_loader = get_train_loader(data_root=data_root)

    # Model
    model = ConvNet(num_classes=cfg.num_classes).to(
        device, dtype=getattr(torch, cfg.dtype)
    )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    total_step = len(train_loader)
    for epoch in range(cfg.num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, 3, 32, 32).to(
                device, dtype=getattr(torch, cfg.dtype)
            )
            labels = labels.to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                logger.info(
                    f"Epoch [{epoch + 1}/{cfg.num_epochs}], "
                    f"Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}"
                )

    # Save checkpoint
    torch.save(model.state_dict(), f"{checkpoint_dir}/model.ckpt")
    logger.info(f"Checkpoint saved to {checkpoint_dir}/model.ckpt")


if __name__ == "__main__":
    train()
