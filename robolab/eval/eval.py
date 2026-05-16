"""Evaluation module for ConvNet on CIFAR-10."""

import torch

from robolab.configs.parameters import TestingParams
from robolab.data import get_test_loader
from robolab.models import ConvNet
from robolab.utils.helpers import get_device, logger

CIFAR10_CLASSES = 10
CLASS_NAMES = [
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


def evaluate(
    checkpoint_path: str = "checkpoints/model.ckpt", data_root: str = "./data"
) -> dict:
    """Evaluate the trained model and return detailed metrics.

    Args:
        checkpoint_path: Path to the model checkpoint file.
        data_root: Root directory for dataset storage.

    Returns:
        dict: Evaluation metrics including overall and per-class accuracy.
    """

    # Load hyperparameters and device
    params = TestingParams()
    device = get_device()

    # Load model checkpoint
    model = ConvNet(num_classes=CIFAR10_CLASSES)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model.to(device, dtype=getattr(torch, params.dtype))
    model.eval()

    test_loader = get_test_loader(data_root=data_root)

    class_correct = [0] * CIFAR10_CLASSES
    class_total = [0] * CIFAR10_CLASSES
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.reshape(-1, 3, 32, 32).to(
                device, dtype=getattr(torch, params.dtype)
            )
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()

            for i in range(labels.size(0)):
                cls = labels[i]
                class_correct[cls] += c[i].item()
                class_total[cls] += 1

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    per_class_accuracy = [
        100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
        for i in range(CIFAR10_CLASSES)
    ]

    return {
        "overall_accuracy": 100.0 * correct / total if total > 0 else 0.0,
        "total_samples": total,
        "correct_predictions": correct,
        "per_class_accuracy": per_class_accuracy,
    }


if __name__ == "__main__":
    metrics = evaluate()
    logger.info(f"Overall Accuracy: {metrics['overall_accuracy']:.2f}%")
    logger.info(f"Total Samples: {metrics['total_samples']}")
    logger.info(f"Correct Predictions: {metrics['correct_predictions']}")
    logger.info("\nPer-Class Accuracy:")
    for name, acc in zip(CLASS_NAMES, metrics["per_class_accuracy"]):
        logger.info(f"  {name:12s}: {acc:.2f}%")
