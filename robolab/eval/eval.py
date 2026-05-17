import torch
from torch import nn

from robolab.utils import get_device


def evaluate(
    model: nn.Module, data_loader: torch.utils.data.DataLoader, dtype: str
) -> dict:
    """Evaluate the trained model and return detailed metrics.

    Args:
        model (nn.Module): The trained PyTorch model to evaluate.
        data_loader (torch.utils.data.DataLoader): The data loader for the test set.
        dtype (str): Data type for tensor operations.

    Returns:
        dict: Evaluation metrics including overall and per-class accuracy.
    """

    # Initialize the model and move it to the appropriate device
    model.eval()
    device = get_device()

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in data_loader:
            images = images.reshape(-1, 3, 32, 32).to(
                device, dtype=getattr(torch, dtype)
            )
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct, total
