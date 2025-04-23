import torch
import numpy as np
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CNN
import torchsummary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_test_data():
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    print(f"{len(test_dataset)} images")
    return DataLoader(test_dataset, batch_size=1)

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
    accuracy = correct / len(test_loader.dataset)
    print(f"Test Accuracy: {accuracy:.4f}")

def measure_inference_time(model, test_loader, num_samples=10000):
    times = []
    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            if i == num_samples:
                break
            images = images.to(device)
            start = time.time()
            model(images)
            end = time.time()
            times.append((end - start) * 1000)
    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"Average Inference Time: {avg_time:.4f} ms | Standard Deviation: {std_time:.4f} ms")
    print(f"Total Inference Time: {np.sum(times) / 1000 :.4f} s")


def main():
    print(f"Using device {device}")
    test_loader = load_test_data()
    model = CNN().to(device)
    torchsummary.summary(model, (1, 28, 28), device=str(device))
    model.load_state_dict(torch.load("model.pth", map_location=device))
    evaluate(model, test_loader)
    measure_inference_time(model, test_loader)

if __name__ == "__main__":
    main()
