"""Common helper functions."""

import logging
from pathlib import Path

import torch
from rich.logging import RichHandler
from torchvision import transforms

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


class AugmentedDataset(torch.utils.data.Dataset):
    """A dataset wrapper that applies augmentations to the data."""

    def __init__(
        self, dataset: torch.utils.data.Dataset, augmentations: transforms.Compose
    ):
        """Initialize the augmented dataset.

        Args:
            dataset (torch.utils.data.Dataset): The original dataset to wrap.
            augmentations (transforms.Compose): The augmentations to apply.
        """
        self.dataset = dataset
        self.augmentations = augmentations

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, label = self.dataset[idx]
        image = self.augmentations(image)
        return image, label
