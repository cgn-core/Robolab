"""Dataset loading module for CIFAR-10 classification.

This module handles the downloading, preprocessing, and loading of the
CIFAR-10 dataset. It creates both training and test DataLoaders with
appropriate transformations.

The CIFAR-10 dataset consists of 60,000 32x32 color images across 10
classes (6,000 images per class), divided into 50,000 training images
and 10,000 test images.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from constants import batch_size

# CIFAR-10 training dataset with tensor transformation
# Downloads automatically if not present; images normalized to [0, 1] range
train_dataset = torchvision.datasets.CIFAR10(
    root="../../data/", train=True, transform=transforms.ToTensor(), download=True
)

# CIFAR-10 test dataset with tensor transformation
# Uses same images split but without download since training handles it
test_dataset = torchvision.datasets.CIFAR10(
    root="../../data/", train=False, transform=transforms.ToTensor()
)

# Training data loader with shuffling for random batch ordering
# Shuffling helps improve model generalization by providing diverse mini-batches each epoch
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)

# Test data loader without shuffling for consistent evaluation results
# No shuffling ensures reproducible evaluation metrics across runs
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False
)
