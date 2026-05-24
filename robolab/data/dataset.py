from copy import deepcopy

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split

from robolab.configs import cfg

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2470, 0.2435, 0.2616)

train_transforms = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
    ]
)

base_train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, transform=None, download=True
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, transform=test_transforms, download=True
)

generator = torch.Generator().manual_seed(42)
val_size = int(0.1 * len(base_train_dataset))
train_size = len(base_train_dataset) - val_size

train_dataset, val_dataset = random_split(
    base_train_dataset, [train_size, val_size], generator=generator
)

train_dataset = deepcopy(train_dataset)
val_dataset = deepcopy(val_dataset)

train_dataset.dataset.transform = train_transforms
val_dataset.dataset.transform = test_transforms

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=cfg.trainparams.batch_size, shuffle=True, pin_memory=True
)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=cfg.trainparams.batch_size, shuffle=False, pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=cfg.trainparams.batch_size, shuffle=False, pin_memory=True
)
