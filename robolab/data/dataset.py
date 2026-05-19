"""Dataset loading and preprocessing for CIFAR-10.

This module defines data loading pipelines for CIFAR-10, including
training transforms with data augmentation, normalization parameters,
and train/validation/test data split utilities.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split

from robolab.configs import cfg

# Normalization statistics for CIFAR-10: computed from training set mean/std
_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# Training transforms: applies normalization with optional augmentation hooks
# (RandomCrop and RandomHorizontalFlip are commented out for ablation)
train_transforms = transforms.Compose(
    [
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
    ]
)

# Test/validation transforms: same normalization without augmentation
test_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
    ]
)

# Load full CIFAR-10 training set with data augmentation transforms
train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, transform=train_transforms, download=True
)

# Load CIFAR-10 test set with evaluation-only transforms
test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, transform=test_transforms, download=True
)

# Create shuffled training data loader with pinned memory for faster GPU transfer
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=cfg.trainparams.batch_size, shuffle=True, pin_memory=True
)

# Create test data loader with deterministic ordering
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=cfg.trainparams.batch_size, shuffle=False, pin_memory=True
)

# Deterministic split of training set into train/validation subsets
generator = torch.Generator().manual_seed(42)
val_size = int(0.1 * len(train_dataset))
train_size = len(train_dataset) - val_size
_, val_dataset = random_split(
    train_dataset, [train_size, val_size], generator=generator
)

# Create validation data loader with deterministic ordering
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=cfg.trainparams.batch_size, shuffle=False, pin_memory=True
)
