"""Common helper functions."""

import logging
from pathlib import Path

import torch
from rich.logging import RichHandler
from torch import nn

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


def save_checkpoint(model: nn.Module, checkpoint_dir: str) -> str:
    """Save the model checkpoint to the specified directory.

    Args:
        model (nn.Module): The PyTorch model to save.
        checkpoint_dir (str): Directory path where the checkpoint file will be saved.

    Returns:
        str: The full path to the saved checkpoint file.
    """
    checkpoint_path = f"{checkpoint_dir}/model.ckpt"
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    """Load the model checkpoint from the specified file.

    Args:
        model (nn.Module): The PyTorch model to load the checkpoint into.
        checkpoint_path (str): The full path to the checkpoint file.
    """
    device = get_device()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    logger.info(f"Checkpoint loaded from {checkpoint_path}")
