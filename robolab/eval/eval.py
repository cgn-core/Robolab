"""Model evaluation module with comprehensive metric computation.

This module provides functions to evaluate trained PyTorch models
on CIFAR-10 datasets, computing accuracy, F1 score, confusion
matrix, and classification reports using scikit-learn.
"""

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

    Runs the model in inference mode over the provided data loader,
    collects predictions, and computes evaluation metrics.

    Args:
        model: The trained PyTorch model to evaluate.
        data_loader: The data loader for the test or validation set.
        dtype: Data type string for tensor operations
            (e.g., ``"float32"``, ``"float16"``).

    Returns:
        Dictionary containing the following keys:

        - ``"accuracy"``: Overall classification accuracy.
        - ``"f1_score"``: Macro-averaged F1 score.
        - ``"confusion_matrix"``: Confusion matrix as a NumPy array.
        - ``"classification_report"``: Per-class precision/recall/F1 report.
    """
    # Disable gradient computation and set model to inference mode
    model.eval()
    device = get_device()

    # Accumulators for batched predictions and ground-truth labels
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device, dtype=getattr(torch, dtype))
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu())
            all_labels.extend(labels.cpu())

    all_preds = torch.stack(all_preds).numpy()
    all_labels = torch.stack(all_labels).numpy()

    # Compute overall accuracy, macro F1 score, confusion matrix,
    # and classification report
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
