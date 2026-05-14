"""Hyperparameters configuration for model training."""

from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters configuration for model training.

    Attributes:
        batch_size: Number of samples per gradient update.
        learning_rate: Step size for optimizer updates.
        num_epochs: Total number of training passes through the dataset.
    """

    batch_size: int = 100
    learning_rate: float = 1e-3
    num_epochs: int = 10
