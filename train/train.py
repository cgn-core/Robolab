import torch
import torch.nn as nn
from datasets.cifar10.dataset import train_loader
from model.model import ConvNet2
from constants import learning_rate, num_epochs

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = ConvNet2(num_classes=10)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# Train the model
def train():
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, 3, 32, 32).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, num_epochs, i + 1, total_step, loss.item()
                    )
                )

    torch.save(model.state_dict(), "checkpoints/model.ckpt")


if __name__ == "__main__":
    train()
