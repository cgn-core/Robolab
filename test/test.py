"""Testing module for evaluating the trained ConvNet model on CIFAR-10.

This module loads the trained model checkpoint and evaluates its accuracy
on the CIFAR-10 test set (10,000 images). It prints the overall test
accuracy as a percentage.

Usage:
    python test/test.py
"""

import torch
from datasets.cifar10.dataset import test_loader
from model.model import ConvNet

# Device configuration - uses GPU if available, otherwise falls back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model instantiation and checkpoint loading
# Creates a new model instance and loads the trained weights from the saved checkpoint
model = ConvNet(num_classes=10)
model.load_state_dict(torch.load("checkpoints/model.ckpt"))
model.to(device)


def test() -> None:
    """Evaluate the trained model on the CIFAR-10 test set.

    Sets the model to evaluation mode (disables dropout), iterates through
    the test data loader, computes predictions, and calculates the overall
    accuracy percentage across all test images.

    Prints:
        Test accuracy percentage formatted as a string.
        Example: "Test Accuracy of the model on the 10000 test images: 75.23 %"

    Returns:
        None
    """
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 3, 32, 32).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(
            "Test Accuracy of the model on the 10000 test images: {} %".format(
                100 * correct / total
            )
        )


if __name__ == "__main__":
    test()
