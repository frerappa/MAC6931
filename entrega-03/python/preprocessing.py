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
import pandas as pd
from utils import load_dataset


def show_images_by_class(dataset, labels: dict, n_images=5):
    n_classes = len(labels)
    class_images = {i: [] for i in range(n_classes)}

    for img, label in dataset:
        if len(class_images[label]) < n_images:
            class_images[label].append(img)
        if all(len(images) == n_images for images in class_images.values()):
            break

    fig, axes = plt.subplots(n_images, n_classes, figsize=(n_classes * 2, n_images * 2,))

    for label, images in class_images.items():
        for i in range(n_images):
            ax = axes[i, label]
            ax.imshow(images[i].squeeze(), cmap="gray")
            ax.axis('off')
            if i == 0:
                ax.set_title(f"{labels[label]}", size='large')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

labels_map = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
}

def main():
    trainset, _, _ = load_dataset()
    show_images_by_class(trainset, labels_map, 6)

    df = pd.read_csv("./csv/power_log.csv")
    df['time_ms'] = df.index * 50
    plt.figure(figsize=(14, 5))
    plt.plot(df['time_ms'], df['power.draw [W]'], label='Consumo (W)')
    plt.xlabel('Tempo (ms)')
    plt.ylabel('Consumo energético (W)')
    plt.title('Consumo de Energia da GPU')
    plt.grid(True)
    plt.savefig("./images/graph.png")

if __name__ == "__main__":
    main()

