"""Dataset loading and preprocessing for CIFAR-10."""

import torch
import torchvision
import torchvision.transforms as transforms

from robolab.configs import TrainingParams


def _get_train_transform() -> transforms.Compose:
    """Training transforms with augmentation for CIFAR-10."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )


def _get_test_transform() -> transforms.Compose:
    """Testing transforms for CIFAR-10."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )


def get_train_loader(
    batch_size: int | None = None, data_root: str = "./data"
) -> torch.utils.data.DataLoader:
    """Create and return the training data loader.

    Args:
        batch_size: Number of samples per batch. Defaults to Hyperparameters.batch_size.
        data_root: Root directory to store/download datasets.

    Returns:
        torch.utils.data.DataLoader: Shuffled training data loader.
    """
    hp = TrainingParams()
    bs = batch_size or hp.batch_size

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, transform=_get_train_transform(), download=True
    )

    return torch.utils.data.DataLoader(
        train_dataset, batch_size=bs, shuffle=True, pin_memory=True
    )


def get_test_loader(
    batch_size: int | None = None, data_root: str = "./data"
) -> torch.utils.data.DataLoader:
    """Create and return the test data loader.

    Args:
        batch_size: Number of samples per batch. Defaults to Hyperparameters.batch_size.
        data_root: Root directory to store/download datasets.

    Returns:
        torch.utils.data.DataLoader: Unshuffled test data loader.
    """
    hp = TrainingParams()
    bs = batch_size or hp.batch_size

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, transform=_get_test_transform()
    )

    return torch.utils.data.DataLoader(
        test_dataset, batch_size=bs, shuffle=False, pin_memory=True
    )
