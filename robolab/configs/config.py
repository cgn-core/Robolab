"""Hyperparameters configuration for model training."""

from typing import Literal

from pydantic import BaseModel, Field


class Hyperparameters(BaseModel):
    """Base class for hyperparameters configuration.

    Attributes:
        logging_level: Logging level (e.g., "INFO", "DEBUG").
        checkpoint_dir: Directory path for saving model checkpoints.
        random_seed: Random seed for reproducibility, or None to disable.
    """

    logging_level: Literal["INFO", "DEBUG", "WARNING", "ERROR"] = "INFO"
    checkpoint_dir: str = "checkpoints"
    random_seed: int | None = 42
    num_classes: int = Field(ge=1, default=10)
    early_stopping_patience: int = Field(ge=1, default=5)


class TrainingParams(BaseModel):
    """Hyperparameters configuration for model training.

    Attributes:
        batch_size: Number of samples per gradient update.
        learning_rate: Step size for optimizer updates.
        num_epochs: Total number of training passes through the dataset.
        dtype: Data type for the model tensors.
    """

    batch_size: int = 100
    learning_rate: float = Field(gt=0.0, default=1e-3)
    num_epochs: int = Field(ge=1, default=10)
    dtype: Literal["float32", "float16", "float64", "int32", "int64"] = "float32"


class TestingParams(BaseModel):
    """Hyperparameters configuration for model testing/evaluation.

    Attributes:
        dtype: Data type for the model tensors during evaluation.
    """

    dtype: Literal["float32", "float16", "float64", "int32", "int64"] = "float32"
