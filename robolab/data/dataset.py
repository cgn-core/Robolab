"""Dataset loading and preprocessing for CIFAR-10."""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split

from robolab.configs import cfg

# Training transforms with data augmentation
train_transforms = transforms.Compose(
    [
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]
)

# Test/validation transforms (no augmentation)
test_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]
)

# Load CIFAR-10 datasets
train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, transform=train_transforms, download=True
)

# Load test dataset with test transforms
test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, transform=test_transforms, download=True
)

# Create data loaders for training and testing
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=cfg.trainparams.batch_size, shuffle=True, pin_memory=True
)

# Create test data loader
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=cfg.trainparams.batch_size, shuffle=False, pin_memory=True
)

generator = torch.Generator().manual_seed(42)

val_size = int(0.1 * len(train_dataset))
train_size = len(train_dataset) - val_size
_, val_dataset = random_split(
    train_dataset, [train_size, val_size], generator=generator
)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=cfg.trainparams.batch_size, shuffle=False, pin_memory=True
)
