import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Define the RankNet model
class RankNet(nn.Module):
    def __init__(self, input_size, hidden_sizes=[64, 32]):
        super(RankNet, self).__init__()
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        layers.append(nn.Linear(prev_size, 1))  # Output layer
        self.model = nn.Sequential(*layers)

    def forward(self, data,mode):
        
        return self.model(x)

# Define the RankNet loss function


# Custom Dataset for pairs of documents
