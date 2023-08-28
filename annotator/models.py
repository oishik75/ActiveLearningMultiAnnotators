import torch
from torch import nn

class AnnotatorModel(nn.Module):
    def __init__(self, n_features, n_annotators) -> None:
        super().__init__()

        self.fc1 = nn.Linear(n_features, 32)
        self.ln1 = nn.LayerNorm(32)
        self.fc2 = nn.Linear(32, 32)
        self.ln2 = nn.LayerNorm(32)
        self.out = nn.Linear(32, n_annotators)

    def forward(self, x):
        out = torch.relu(self.ln1(self.fc1(x)))
        out = torch.relu(self.ln2(self.fc2(out)))
        out = torch.sigmoid(self.out(out))

        return out