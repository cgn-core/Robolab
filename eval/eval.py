"""Evaluation module for the ConvNet model on CIFAR-10.

This module provides utilities for evaluating the trained model's performance
on the CIFAR-10 dataset. It can be extended to include detailed metrics
such as per-class accuracy, confusion matrix, and classification reports.

Usage:
    python eval/eval.py
"""

import torch
from datasets.cifar10.dataset import test_loader
from model.model import ConvNet


def evaluate_model(checkpoint_path: str = "checkpoints/model.ckpt") -> dict:
    """Evaluate the trained model and return detailed metrics.

    Loads the model checkpoint, runs inference on the test set,
    and computes overall accuracy along with per-class metrics.

    Args:
        checkpoint_path (str): Path to the model checkpoint file.
            Defaults to "checkpoints/model.ckpt".

    Returns:
        dict: A dictionary containing evaluation metrics including:
            - 'overall_accuracy': Overall classification accuracy (float)
            - 'total_samples': Total number of test samples (int)
            - 'correct_predictions': Number of correct predictions (int)
    """
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model instantiation and checkpoint loading
    model = ConvNet(num_classes=10)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    class_correct = list(0 for _ in range(10))
    class_total = list(0 for _ in range(10))

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.reshape(-1, 3, 32, 32).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == labels).squeeze()

            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Compute per-class accuracy
    class_accuracy = []
    for i in range(10):
        if class_total[i] > 0:
            class_accuracy.append(100 * class_correct[i] / class_total[i])
        else:
            class_accuracy.append(0.0)

    metrics = {
        "overall_accuracy": 100 * correct / total,
        "total_samples": total,
        "correct_predictions": correct,
        "per_class_accuracy": class_accuracy,
    }

    return metrics


if __name__ == "__main__":
    metrics = evaluate_model()
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.2f}%")
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"Correct Predictions: {metrics['correct_predictions']}")
    print("\nPer-Class Accuracy:")
    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    for name, acc in zip(class_names, metrics["per_class_accuracy"]):
        print(f"  {name:12s}: {acc:.2f}%")
