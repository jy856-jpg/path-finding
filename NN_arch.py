import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import math
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader


class LSTM_sMNIST(nn.Module):
    def __init__(self, hidden_size=48, num_classes=10, seed=1234, perm=None):
        super().__init__()
        # Fix permutation for reproducibility
        g = torch.Generator().manual_seed(seed)
        if perm is None:
            self.perm = torch.randperm(28*28, generator=g)  # perm over all 784 pixels
        else:
            self.perm = perm

        # Each timestep = 28 features (row of image)
        self.gru = nn.LSTM(input_size=28, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes, bias=False)

    def forward(self, x):
    # x: (batch, 1, 28, 28)
        x = x.view(x.size(0), -1)             # flatten -> (batch, 784)
       # perm = self.perm.to(x.device)         # ensure perm is on same device as x
        x = x[:, self.perm]                        # apply permutation
        x = x.view(x.size(0), 28, 28)         # reshape back -> (batch, 28, 28)
        out, _ = self.gru(x)                  # process 28 timesteps
        out = out[:, -1, :]                   # last timestep hidden state
        out = self.fc(out)                     # classifier
        return out
    
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)  # Output: 28x28x6
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # Output: 14x14x6
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)  # Output: 10x10x16
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # Output: 5x5x16
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10 output classes (like MNIST)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.shape[0], -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class FCP(nn.Module):
    def __init__(self):
        super(FCP, self).__init__()
        self.fc1 = nn.Linear(28*28, 50,bias=False)  # Input layer to hidden layer 1
        self.fc2 = nn.Linear(50, 50,bias=False)     # Hidden layer 1 to hidden layer 2
        self.fc3 = nn.Linear(50, 10,bias=False)      # Hidden layer 2 to output layer

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input
        x = F.softplus(self.fc1(x))  # Apply ReLU activation after first layer
        x = F.softplus(self.fc2(x))  # Apply ReLU activation after second layer
        x = self.fc3(x)          # Output layer (no activation function)
        return x
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
        nn.Linear(784, 32,bias=False),
        nn.Softplus(),
        nn.Linear(32, 32,bias=False),
        nn.Softplus(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 32,bias=False),
            nn.Softplus(),
            nn.Linear(32, 784,bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x