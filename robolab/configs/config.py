"""Hyperparameters configuration for model training."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class Hyperparameters(BaseModel):
    """Hyperparameters configuration.

    Attributes:
        logging_level: Logging level (e.g., "INFO", "DEBUG").
        checkpoint_dir: Directory path for saving model checkpoints.
        random_seed: Random seed for reproducibility, or None to disable.
        num_classes: Number of output classification classes.
        early_stopping_patience: Epochs to wait for validation improvement.
    """

    logging_level: Literal["INFO", "DEBUG", "WARNING", "ERROR"] = "INFO"
    checkpoint_dir: str = "checkpoints"
    random_seed: int | None = 42
    num_classes: int = Field(ge=1, default=10)
    early_stopping_patience: int = Field(ge=1, default=5)


class TrainingParams(BaseModel):
    """Training hyperparameters configuration.

    Attributes:
        batch_size: Number of samples per gradient update.
        learning_rate: Step size for optimizer updates.
        num_epochs: Total number of training passes through the dataset.
        dtype: Data type for the model tensors.
        weight_decay: L2 regularization factor.
        warmup_epochs: Number of warmup epochs for learning rate scheduler.
        max_grad_norm: Maximum gradient norm for clipping.
        class_weights: Optional class weights for imbalanced datasets.
    """

    batch_size: int = 100
    weight_decay: float = Field(gt=0.0, default=1e-4)
    warmup_epochs: int = Field(ge=0, default=5)
    max_grad_norm: float = Field(gt=0.0, default=1.0)
    class_weights: list[float] | None = None
    learning_rate: float = Field(gt=0.0, default=1e-3)
    num_epochs: int = Field(ge=1, default=50)
    dtype: Literal["float32", "float16", "float64", "int32", "int64"] = "float32"


class TestingParams(BaseModel):
    """Testing/evaluation hyperparameters configuration.

    Attributes:
        dtype: Data type for the model tensors during evaluation.
    """

    dtype: Literal["float32", "float16", "float64", "int32", "int64"] = "float32"


class Config(BaseModel):
    """Application configuration loaded from YAML.

    Attributes:
        hyperparams: Common hyperparameters shared across modules.
        trainparams: Training-specific hyperparameters.
        testparams: Testing/evaluation-specific hyperparameters.

    Example:
        Load from YAML (global singleton):

        >>> cfg = Config.load_from_yaml()

        Load from dict (useful for tests):

        >>> cfg = Config(
        ...     hyperparams=Hyperparameters(num_classes=10),
        ...     trainparams=TrainingParams(),
        ...     testparams=TestingParams(),
        ... )
    """

    hyperparams: Hyperparameters = Field(default_factory=Hyperparameters)
    trainparams: TrainingParams = Field(default_factory=TrainingParams)
    testparams: TestingParams = Field(default_factory=TestingParams)

    @classmethod
    def load_from_yaml(cls, path: str | None = None) -> "Config":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML config file.
                  Defaults to ``config.yaml`` in the same directory as this module.

        Returns:
            Config: An instance of the configuration.

        Raises:
            yaml.YAMLError: If the YAML file is invalid.
            ValidationError: If the configuration fails Pydantic validation.
        """
        config_path = Path(path) if path else Path(__file__).parent / "config.yaml"
        with open(config_path, encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)


# Global singleton for backward compatibility.
# Prefer explicit ``Config`` injection in new code.
cfg: Config = Config.load_from_yaml()
