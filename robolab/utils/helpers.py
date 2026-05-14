"""Common helper functions."""

import torch


def get_device() -> torch.device:
    """Get the available compute device (CUDA if available, else CPU).

    Returns:
        torch.device: The device to use for computation.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
