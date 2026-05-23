"""Test module for evaluating ConvNet on CIFAR-10 test set.

This module loads a trained model checkpoint and evaluates it on the
full CIFAR-10 test set, logging overall and per-class metrics.
"""

from robolab.configs import cfg
from robolab.data import test_loader
from robolab.eval import evaluate
from robolab.models import ResNet18
from robolab.utils import (
    get_device,
    load_checkpoint,
    logger,
)


def test(
    checkpoint_path: str = "checkpoints/model.ckpt",
    dtype: str = "float32",
) -> dict:
    """Evaluate the trained model on the full CIFAR-10 test set.

    Loads a model checkpoint from disk, runs inference over the test
    loader, and returns a dictionary of aggregated metrics.

    Args:
        checkpoint_path: File path to the saved model checkpoint.
        dtype: Data type string for tensor operations
            (e.g., ``"float32"``, ``"float16"``).

    Returns:
        dict: Evaluation metrics including overall accuracy, F1 score,
            confusion matrix, and classification report.
    """
    logger.info("Starting evaluation on the test set...")

    # Determine compute device (CUDA if available, otherwise CPU)
    device = get_device()

    # Instantiate model architecture and restore weights from checkpoint
    model = ResNet18(num_classes=cfg.hyperparams.num_classes).to(device)
    load_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
    )

    # Run comprehensive evaluation via the eval module
    metrics = evaluate(model, test_loader, dtype)

    # Aggregate evaluation results into a single metrics dictionary
    metrics = {
        "overall_accuracy": metrics["accuracy"],
        "total_samples": len(test_loader.dataset),
        "correct_predictions": int(metrics["accuracy"] * len(test_loader.dataset)),
        "f1_score": metrics["f1_score"],
        "confusion_matrix": metrics["confusion_matrix"],
        "classification_report": metrics["classification_report"],
    }

    # Log key evaluation results to console and file
    logger.info(f"Total Test Samples: {metrics['total_samples']}")
    logger.info(f"Correct Predictions: {metrics['correct_predictions']}")
    logger.info(f"Overall Test Accuracy: {metrics['overall_accuracy'] * 100:.2f} %")
    logger.info(f"F1 Score: {metrics['f1_score'] * 100:.4f} %")

    return metrics


if __name__ == "__main__":
    test()
