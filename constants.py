from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters configuration for model training.

    This dataclass holds the key hyperparameters used during the
    training process, including batch size, learning rate, and
    number of epochs.

    Attributes:
        batch_size: Number of samples per gradient update.
            Default is 100.
        learning_rate: Step size for optimizer updates.
            Default is 0.001.
        num_epochs: Total number of training passes through
            the full training dataset.
            Default is 10.
    """

    batch_size: int = 100
    learning_rate: float = 0.001
    num_epochs: int = 10
