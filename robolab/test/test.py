"""Test module for evaluating ConvNet on CIFAR-10 test set."""

from robolab.eval import evaluate
from robolab.utils import logger, total_params


def test(checkpoint_path: str = "checkpoints/model.ckpt") -> None:
    """Evaluate the trained model on the CIFAR-10 test set.

    Args:
        checkpoint_path: Path to the model checkpoint file.
    """
    logger.info("Starting evaluation on the test set...")
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    metrics = evaluate(checkpoint_path=checkpoint_path)
    logger.info(
        f"Test Accuracy of the model on the {metrics['total_samples']} test images: "
        f"{metrics['overall_accuracy']:.2f} %"
    )


if __name__ == "__main__":
    test()
