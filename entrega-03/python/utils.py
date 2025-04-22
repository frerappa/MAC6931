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

def load_dataset():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_val = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_size = int(0.8 * len(train_val))
    val_size = len(train_val) - train_size
    train, val = random_split(train_val, [train_size, val_size])

    return train, val, test