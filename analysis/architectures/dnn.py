import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sklearn
from sklearn.metrics import roc_auc_score 



# A (very) simple DNN to get started 
class DNN(nn.Module):
    def __init__(self, input_size, hidden_size = 100, dropout = 0.05):
        super().__init__()
        self.flatten = nn.Flatten()
        self.bn = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.bn(x)

        x = F.relu(self.fc1(x))

        x = self.dropout1(x)
        x = F.relu(self.fc2(x))

        x = self.dropout2(x)
        x = self.fc3(x)

        return x