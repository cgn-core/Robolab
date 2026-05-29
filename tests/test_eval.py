import pytest
import torch

from src.data import test_loader
from src.eval import evaluate
from src.models import model_factory


@pytest.fixture
def model():
    return model_factory(num_classes=10)


def test_confusion_matrix(model):
    results = evaluate(model, torch.device("cuda"), test_loader, dtype=torch.float16)
    cm = results["confusion_matrix"]
    print(cm)
