import torch
from torch import nn

class NeuralNetClassifier(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()

        self.fc1 = nn.Linear(n_features, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        self.out = nn.Linear(256, n_classes)

    def forward(self, x):
        out = torch.relu(self.ln1(self.fc1(x)))
        out = torch.relu(self.ln2(self.fc2(out)))
        out = self.out(out)

        return out