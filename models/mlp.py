from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dims, expansion_factor = 2, dropout = 0.1):
        super().__init__()
        num_hidden = input_dims * expansion_factor
        self.fc1 = nn.Linear(input_dims, num_hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(num_hidden, input_dims)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout1(F.gelu(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        return x


class MLPEncoder(nn.Module):
    def __init__(self, input_dims, expansion_factor, output_dims, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(input_dims)
        self.mlp = MLP(input_dims, expansion_factor, dropout=0.1)
        self.linear = nn.Linear(input_dims, output_dims)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mlp(x)
        x = x + residual

        x = self.linear(x)

        return x

