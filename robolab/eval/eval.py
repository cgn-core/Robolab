import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
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

    # Initialize lists to store predictions and true labels
    all_preds, all_labels = [], []

    # Evaluate the model on the test set
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device, dtype=getattr(torch, dtype))
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu())
            all_labels.extend(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Calculate overall accuracy, F1 score, confusion matrix, and classification report
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds)
    cr = classification_report(all_labels, all_preds)

    return {
        "accuracy": acc,
        "f1_score": f1,
        "confusion_matrix": cm,
        "classification_report": cr,
    }
