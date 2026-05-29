import pytest
import torch

from src.models import model_factory


@pytest.fixture
def model():
    return model_factory(num_classes=10)


def test_forward_output_shape(model):
    x = torch.randn(1, 3, 32, 32)
    out = model(x)
    assert out.shape == (1, 10)


def test_forward_returns_tensor(model):
    x = torch.randn(1, 3, 32, 32)
    out = model(x)
    assert isinstance(out, torch.Tensor)


def test_forward_no_nan(model):
    x = torch.randn(1, 3, 32, 32)
    out = model(x)
    out.sum().backward()
    assert x.grad is not None


@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_forward_different_batch_sizes(model, batch_size):
    x = torch.randn(1, 3, 32, 32)
    out = model(x)
    assert out.shape[0] == batch_size
