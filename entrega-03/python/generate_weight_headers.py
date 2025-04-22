import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import pickle
import time
import os
import math
import matplotlib.pyplot as plt
from model import CNN
from utils import load_dataset

# Constants
input_size = (28, 28)
conv_kernel_size = (7, 7)
conv_filter_num = 4
pool_size = (2, 2)
dense_size = 10
training_epochs = 6
pad = conv_kernel_size[0] // 2

# Paths
headers_dir = "../headers"
os.makedirs(headers_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





def train_model(model, train_loader, val_loader):
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

    for epoch in range(training_epochs):
        model.train()
        correct, total, epoch_loss = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        loss_avg = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        correct, val_loss = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                pred = outputs.argmax(dim=1)
                correct += (pred == labels).sum().item()

        val_acc = correct / len(val_loader.dataset)
        val_loss_avg = val_loss / len(val_loader)

        history['accuracy'].append(acc)
        history['val_accuracy'].append(val_acc)
        history['loss'].append(loss_avg)
        history['val_loss'].append(val_loss_avg)

        print(f"Epoch {epoch+1}: Train Acc = {acc:.4f}, Val Acc = {val_acc:.4f}")

    return history

def plot_history(history):
    plt.plot(history['accuracy'], label='Train Acc')
    plt.plot(history['val_accuracy'], label='Val Acc')
    plt.title('Model Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig('history_accuracy.png')
    plt.clf()

    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.grid()
    plt.savefig('history_loss.png')
    plt.clf()

def write_header_definitions():
    with open(f"{headers_dir}/definitions.h", 'w') as f:
        f.write(f"""#pragma once
#define DIGITS 10
#define IMG_ROWS {input_size[0]}
#define IMG_COLS {input_size[1]}

// Padding
#define KRN_ROWS {conv_kernel_size[0]}
#define KRN_COLS {conv_kernel_size[1]}
#define FILTERS {conv_filter_num}
#define PAD_ROWS (KRN_ROWS - 1)
#define PAD_COLS (KRN_COLS - 1)
#define PAD_IMG_ROWS (IMG_ROWS + PAD_ROWS)
#define PAD_IMG_COLS (IMG_COLS + PAD_COLS)

// Pool layer
#define POOL_ROWS {pool_size[0]}
#define POOL_COLS {pool_size[1]}
#define POOL_IMG_ROWS (IMG_ROWS / POOL_ROWS)
#define POOL_IMG_COLS (IMG_COLS / POOL_COLS)

// Flatten layer
#define FLAT_SIZE (FILTERS * POOL_IMG_ROWS * POOL_IMG_COLS)

// Dense layer
#define DENSE_SIZE {dense_size}
""")

def write_conv_weights(model):
    weights = model.conv.weight.detach().cpu().numpy()
    biases = model.conv.bias.detach().cpu().numpy()

    with open(f"{headers_dir}/conv_weights.h", 'w') as f:
        f.write("#pragma once\n#include \"definitions.h\"\n\n")

        f.write(f"// Conv layer weights\nfloat conv_weights[FILTERS][KRN_ROWS][KRN_COLS] = {{\n")
        for f_idx in range(conv_filter_num):
            f.write("    {\n")
            for r in range(conv_kernel_size[0]):
                row = ", ".join(str(weights[f_idx][0][r][c]) for c in range(conv_kernel_size[1]))
                f.write(f"        {{ {row} }}")
                f.write(",\n" if r != conv_kernel_size[0] - 1 else "\n")
            f.write("    }" + (",\n" if f_idx != conv_filter_num - 1 else "\n"))
        f.write("};\n\n")

        f.write("// Conv layer biases\nfloat conv_biases[FILTERS] = { " +
                ", ".join(str(b) for b in biases) + " };\n")

def write_dense_weights(model):
    weights = model.fc.weight.detach().cpu().numpy()
    biases = model.fc.bias.detach().cpu().numpy()

    with open(f"{headers_dir}/dense_weights.h", 'w') as f:
        f.write("#pragma once\n#include \"definitions.h\"\n\n")

        f.write(f"// Dense layer weights\nfloat dense_weights[FLAT_SIZE][DENSE_SIZE] = {{\n")
        for i in range(weights.shape[0]):
            row = ", ".join(str(weights[i][j]) for j in range(weights.shape[1]))
            f.write(f"    {{ {row} }}")
            f.write(",\n" if i != weights.shape[0] - 1 else "\n")
        f.write("};\n\n")

        f.write("// Dense layer biases\nfloat dense_biases[DENSE_SIZE] = { " +
                ", ".join(str(b) for b in biases) + " };\n")

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
    acc = correct / len(test_loader.dataset)
    print(f"Test Accuracy: {acc:.3f}")

def measure_prediction_time(model, test_loader):
    times = []
    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(test_loader):
            if i == 100: break
            x = x.to(device)
            start = time.time()
            model(x)
            end = time.time()
            times.append((end - start) * 1000)
    mean_time = np.mean(times)
    print(f"Mean time per prediction: {mean_time:.4f} ms")

def main():
    train_ds, val_ds, test_ds = load_dataset()
    print(f"Training images: {len(train_ds)} | Validation images: {len(val_ds)} | Testing images: {len(test_ds)}")
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    test_loader = DataLoader(test_ds, batch_size=1)

    model = CNN().to(device)
    history = train_model(model, train_loader, val_loader)
    torch.save(model.state_dict(), "model.pth")
    pickle.dump(history, open("train_history_dict", "wb"))

    plot_history(history)
    evaluate(model, test_loader)
    measure_prediction_time(model, test_loader)

    write_header_definitions()
    write_conv_weights(model)
    write_dense_weights(model)

if __name__ == "__main__":
    main()
