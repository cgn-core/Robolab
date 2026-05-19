"""Training script for ConvNet on CIFAR-10."""

import torch
import torch.amp as amp
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.utils.tensorboard import SummaryWriter

from robolab.configs import cfg
from robolab.data import train_loader, val_loader
from robolab.eval import evaluate
from robolab.models import ConvNet
from robolab.utils import get_device, logger, save_checkpoint, total_params


class EarlyStopping:
    """Early stopping utility to prevent overfitting.

    Saves the best model checkpoint based on validation accuracy.
    Stops training when accuracy hasn't improved for 'patience' epochs.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        """
        Args:
            patience: Number of epochs to wait for improvement before stopping.
            min_delta: Minimum change in the monitored metric to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.checkpoint_path = None

    def __call__(self, model: nn.Module, score: float, checkpoint_dir: str) -> None:
        """
        Args:
            model: The model to save checkpoints for.
            score: The validation accuracy score.
            checkpoint_dir: Directory to save checkpoints.
        """
        if self.best_score is None:
            self.best_score = score
            self.checkpoint_path = save_checkpoint(
                model=model, checkpoint_dir=checkpoint_dir
            )
            logger.info(f"New best validation accuracy: {score * 100:.2f}%")

        elif score > self.best_score + self.min_delta:
            self.best_score = score
            self.checkpoint_path = save_checkpoint(
                model=model, checkpoint_dir=checkpoint_dir
            )
            self.counter = 0
            logger.info(f"New best validation accuracy: {score * 100:.2f}%")

        else:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info("Early stopping triggered.")


def train(checkpoint_dir: str = "checkpoints", data_root: str = "./data") -> None:
    """Train the ConvNet model on CIFAR-10.

    Args:
        checkpoint_dir: Directory to save model checkpoints.
        data_root: Root directory for dataset storage.
    """

    # Create fresh early stopping instance for each training run
    early_stopping = EarlyStopping(patience=cfg.hyperparams.early_stopping_patience)
    device = get_device()

    # Set random seed for reproducibility
    if cfg.hyperparams.random_seed is not None:
        torch.manual_seed(cfg.hyperparams.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.hyperparams.random_seed)

    # Model
    model = ConvNet(num_classes=cfg.hyperparams.num_classes).to(device)
    logger.info(f"total_params: {total_params(model):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.trainparams.learning_rate,
        weight_decay=cfg.trainparams.weight_decay,
    )

    # Cosine annealing scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Gradient clipping
    max_grad_norm = 1.0
    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    # Mixed precision training setup
    if device.type == "cuda":
        scaler = amp.GradScaler(device)
    else:
        scaler = None  # No need for scaler on CPU

    # TensorBoard writer
    writer = SummaryWriter(log_dir="runs/convnet_cifar10")

    total_step = len(train_loader)
    for epoch in range(cfg.trainparams.num_epochs):
        # Training loop
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with amp.autocast(device.type, dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if (i + 1) % 100 == 0:
                writer.add_scalar("Training Loss", loss.item(), epoch * total_step + i)
                logger.info(
                    f"Epoch [{epoch + 1}/{cfg.trainparams.num_epochs}], "
                    f"Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}"
                )

        # Validation
        model.eval()
        metrics = evaluate(model, val_loader, dtype=cfg.trainparams.dtype)

        val_accuracy = metrics["accuracy"]
        writer.add_scalar("Validation Accuracy", val_accuracy, epoch)
        logger.info(
            f"Validation Accuracy of the model on the "
            f"{len(val_loader.dataset)} validation images: {val_accuracy:.2f} %"
        )

        # Step scheduler after each epoch (CosineAnnealingWarmRestarts is epoch-based)
        scheduler.step()

        # Check for early stopping
        early_stopping(model, val_accuracy, checkpoint_dir)
        if early_stopping.early_stop:
            break


if __name__ == "__main__":
    train()
