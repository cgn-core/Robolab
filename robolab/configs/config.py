"""Hyperparameters configuration for model training."""

import difflib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from robolab.utils import logger


class Hyperparameters(BaseModel):
    """Hyperparameters configuration.

    Attributes:
        logging_level: Logging level (e.g., "INFO", "DEBUG").
        checkpoint_dir: Directory path for saving model checkpoints.
        random_seed: Random seed for reproducibility, or None to disable.
        num_classes: Number of output classification classes.
        early_stopping_patience: Epochs to wait for validation improvement.
    """

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

    batch_size: int = 100
    weight_decay: float = Field(gt=0.0, default=1e-4)
    warmup_epochs: int = Field(ge=0, default=5)
    max_grad_norm: float = Field(gt=0.0, default=1.0)
    class_weights: list[float] | None = None
    learning_rate: float = Field(gt=0.0, default=1e-3)
    num_epochs: int = Field(ge=1, default=50)
    criterion: Literal["CrossEntropyLoss", "FocalLoss", "LabelSmoothing", "MSE"] = (
        "CrossEntropyLoss"
    )
    optimizer: Literal["Adam", "AdamW", "RMSProp", "SGD"] = "AdamW"
    dtype: Literal["float32", "float16", "float64", "int32", "int64"] = "float32"


class TestingParams(BaseModel):
    """Testing/evaluation hyperparameters configuration.

    Attributes:
        dtype: Data type for the model tensors during evaluation.
    """

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

    hyperparams: Hyperparameters = Field(default_factory=Hyperparameters)
    trainparams: TrainingParams = Field(default_factory=TrainingParams)
    testparams: TestingParams = Field(default_factory=TestingParams)

    @classmethod
    def load_from_yaml(cls, path: str | None = None) -> "Config":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML config file.
                  Defaults to ``config.yaml`` in the same directory as this module.
                  If ``None`` or not provided, the default location is used.

        Returns:
            Config: An instance of the configuration loaded from the YAML file.

        Raises:
            FileNotFoundError: If the YAML file does not exist at the specified path.
            yaml.YAMLError: If the YAML file is invalid or cannot be parsed.
            ValidationError: If the configuration fails Pydantic validation.
        """

        try:
            with open(
                path or (Path(__file__).parent / "config.yaml"), encoding="utf-8"
            ) as f:
                config_data = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at: {path}")
        try:
            return cls(**config_data)
        except ValidationError as e:
            # Tüm doğrulama hatalarını topla
            errors = e.errors()

            for error in errors:
                # Pydantic v2'de loc bir dict olabilir, tuple olarak ele al
                loc_list = error.get("loc", [])
                location = " -> ".join(str(loc) for loc in loc_list)
                wrong_field = loc_list[-1] if loc_list else "unknown"
                error_type = error.get("type", "unknown")

                logger.error(
                    f"Validation error at '{location}': {error['msg']} "
                    f"(field: '{wrong_field}')"
                )

                if error_type == "extra_forbidden":
                    logger.error(
                        f"Unexpected field '{wrong_field}' found "
                        "in configuration. Please remove it."
                    )
                    # Suggested fields from the parent model
                    parent_model_path = loc_list[:-1]
                    if parent_model_path:
                        current_model = cls
                        for loc in parent_model_path:
                            if (
                                hasattr(current_model, "model_fields")
                                and loc in current_model.model_fields
                            ):
                                field_type = current_model.model_fields[loc].annotation
                                # Handle AnnotationType like "Hyperparameters" or "TrainingParams"
                                if hasattr(field_type, "__annotations__"):
                                    current_model = field_type
                                elif hasattr(field_type, "__origin__"):
                                    # Handle Union types like str | None
                                    if hasattr(field_type, "__args__"):
                                        for arg in field_type.__args__:
                                            if hasattr(arg, "__annotations__"):
                                                current_model = arg
                                                break
                        if hasattr(current_model, "model_fields"):
                            valid_fields = list(current_model.model_fields.keys())
                            suggestions = difflib.get_close_matches(
                                wrong_field,
                                valid_fields,
                                n=3,
                                cutoff=0.3,
                            )
                            if suggestions:
                                logger.error(
                                    f"Did you mean one of: {', '.join(repr(s) for s in suggestions)}?"
                                )
                            else:
                                logger.error(
                                    f"No close matches found for '{wrong_field}'. "
                                    f"Valid fields in '{current_model.__name__}' are: {valid_fields}"
                                )

                elif error_type == "value_error":
                    # Value constraint errors (e.g., ge, gt, literal)
                    logger.error(
                        f"Invalid value for '{wrong_field}'. "
                        f"Please check the constraint and try again."
                    )

                elif error_type == "type_error":
                    logger.error(
                        f"Type mismatch for '{wrong_field}'. "
                        f"Please check the expected type."
                    )

                elif error_type == "missing":
                    logger.error(f"Required field '{wrong_field}' is missing.")

            raise ValueError(f"Configuration validation error: {e}")


cfg: Config = Config.load_from_yaml()
