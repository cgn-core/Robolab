# Robolab - CIFAR-10 Image Classification

ConvNet model with PyTorch for CIFAR-10 image classification.

## Setup

```bash
uv sync
```

## Usage

### Training

```bash
uv run python -m robolab.train
```

### Testing

```bash
uv run python -m robolab.test
```

### Evaluation (per-class metrics)

```bash
uv run python -m robolab.eval
```

### TensorBoard
```bash
tensorboard --logdir=runs
```

## Project Structure

```
robolab/
├── configs/       # Hyperparameters and configuration
├── data/          # Dataset loaders and transforms
├── models/        # Model definitions
├── utils/         # Helper functions