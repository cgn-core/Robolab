import torch
from datasets.cifar10.dataset import test_loader
from model.model import ConvNet2

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Model
model = ConvNet2(num_classes=10)
model.load_state_dict(torch.load("checkpoints/model.ckpt"))
model.to(device)

def test():
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