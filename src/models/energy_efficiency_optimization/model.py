# src/models/energy_efficiency_optimization/model.py

import torch
import torch.nn as nn

class EnergyEfficiencyNet(nn.Module):
    def __init__(self, input_dim, hidden=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, x):
        return self.net(x)


'''import torch.nn as nn

class EnergyEfficiencyModel(nn.Module):
    """
    PyTorch model for energy efficiency optimization in 5G OpenRAN.
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        super(EnergyEfficiencyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
'''
