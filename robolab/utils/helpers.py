"""Common helper functions."""

import logging
from pathlib import Path

import torch
from rich.logging import RichHandler

from robolab.configs import Hyperparameters

cfg = Hyperparameters()

current_dir = Path(__file__).resolve().parent
log_dir = current_dir.parent.parent / "logs"
log_file = log_dir / "robolab.log"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=cfg.logging_level,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(rich_tracebacks=True),
        logging.FileHandler(log_file, encoding="utf-8"),
    ],
)
logger = logging.getLogger("robolab")


def get_device() -> torch.device:
    """Get the available compute device.

    Returns:
        torch.device: CUDA if available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def num_trainable_params(model: torch.nn.Module) -> int:
    """Calculate the total number of trainable parameters in the model.

    Args:
        model (torch.nn.Module): The PyTorch model to analyze.

    Returns:
        int: Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def total_params(model: torch.nn.Module) -> int:
    """Calculate the total number of parameters in the model.

    Args:
        model (torch.nn.Module): The PyTorch model to analyze.

    Returns:
        int: Total number of parameters.
    """
    return sum(p.numel() for p in model.parameters())
