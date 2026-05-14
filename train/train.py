"""Training script for the ConvNet model on CIFAR-10 dataset.

This module handles the complete training loop, including data loading,
model instantiation, optimization, and checkpoint saving.

Usage:
    python train/train.py
"""

import torch
import torch.nn as nn
from datasets.cifar10.dataset import train_loader
from model.model import ConvNet
from constants import Hyperparameters

hyperparams = Hyperparameters()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = ConvNet(num_classes=10)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams.learning_rate)


def train() -> None:
    """Train the ConvNet model on the CIFAR-10 dataset.

    Iterates over the training data for the specified number of epochs,
    computing loss and updating model weights using the AdamW optimizer.
    Prints progress every 100 batches and saves the final model checkpoint
    to disk upon completion.

    Returns:
        None
    """
    total_step = len(train_loader)
    for epoch in range(hyperparams.num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, 3, 32, 32).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1,
                        hyperparams.num_epochs,
                        i + 1,
                        total_step,
                        loss.item(),
                    )
                )

    torch.save(model.state_dict(), "checkpoints/model.ckpt")


if __name__ == "__main__":
    train()
