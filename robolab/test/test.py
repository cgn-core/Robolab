"""Test module for evaluating ConvNet on CIFAR-10 test set."""

from robolab.configs import Hyperparameters
from robolab.data import test_loader
from robolab.eval import evaluate
from robolab.models import ConvNet
from robolab.utils import (
    get_device,
    load_checkpoint,
    logger,
)


def test(
    checkpoint_path: str = "checkpoints/model.ckpt",
    dtype: str = "float32",
) -> dict:
    """Evaluate the trained model and return detailed metrics.

    Args:
        checkpoint_path: Path to the model checkpoint file.
        data_root: Root directory for dataset storage.
        dtype: Data type for tensor operations.
    Returns:
        dict: Evaluation metrics including overall and per-class accuracy.
    """

    logger.info("Starting evaluation on the test set...")

    # Load hyperparameters and device
    device = get_device()

    # Load model checkpoint
    model = ConvNet(num_classes=Hyperparameters().num_classes).to(device)
    load_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
    )

    # Initialize the model and move it to the appropriate device
    class_correct = [0] * Hyperparameters().num_classes
    class_total = [0] * Hyperparameters().num_classes

    # Evaluate the model on the test set
    metrics = evaluate(model, test_loader, dtype)

    # Calculate per-class accuracy
    per_class_accuracy = [
        100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
        for i in range(Hyperparameters().num_classes)
    ]

    # Compile metrics into a dictionary
    metrics = {
        "overall_accuracy": metrics["accuracy"],
        "total_samples": len(test_loader.dataset),
        "correct_predictions": int(metrics["accuracy"] * len(test_loader.dataset)),
        "per_class_accuracy": per_class_accuracy,
        "f1_score": metrics["f1_score"],
        "confusion_matrix": metrics["confusion_matrix"],
        "classification_report": metrics["classification_report"],
    }

    # Log the results
    logger.info(f"Total Test Samples: {metrics['total_samples']}")
    logger.info(f"Correct Predictions: {metrics['correct_predictions']}")
    logger.info(f"Overall Test Accuracy: {metrics['overall_accuracy'] * 100:.2f} %")
    logger.info(f"F1 Score: {metrics['f1_score'] * 100:.4f} %")

    return metrics


if __name__ == "__main__":
    test()
