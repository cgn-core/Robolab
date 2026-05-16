"""Hyperparameters configuration for model training."""

from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Base class for hyperparameters configuration.

    Attributes:
        logging_level: Logging level (e.g., "INFO", "DEBUG").
        checkpoint_dir: Directory path for saving model checkpoints.
        random_seed: Random seed for reproducibility, or None to disable.
    """

    logging_level: str = "INFO"
    checkpoint_dir: str = "checkpoints"
    random_seed: int | None = 42


@dataclass
class TrainingParams:
    """Hyperparameters configuration for model training.

    Attributes:
        batch_size: Number of samples per gradient update.
        learning_rate: Step size for optimizer updates.
        num_epochs: Total number of training passes through the dataset.
        dtype: Data type for the model tensors.
    """

    batch_size: int = 100
    learning_rate: float = 1e-3
    num_epochs: int = 10
    dtype: str = "float32"


@dataclass
class TestingParams:
    """Hyperparameters configuration for model testing/evaluation.

    Attributes:
        dtype: Data type for the model tensors during evaluation.
    """

    dtype: str = "float16"
