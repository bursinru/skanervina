import torch
from torch import nn
from torch.nn import functional as F


class QueryAdapter(nn.Module):
    """Small residual query correction; original gallery vectors remain unchanged."""
    def __init__(self, dimension=768, hidden=128):
        super().__init__()
        self.down = nn.Linear(dimension, hidden)
        self.up = nn.Linear(hidden, dimension)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, values):
        return F.normalize(values + self.up(F.gelu(self.down(values))), dim=-1)
