import torch
import torch.nn as nn


class QKVProjection(nn.Module):

    def __init__(self, embedding_dim):

        super().__init__()
        # Separate learnable projection layers
        self.query = nn.Linear(embedding_dim, embedding_dim)

        self.key = nn.Linear(embedding_dim, embedding_dim)

        self.value = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):

        # Generate Query, Key and Value
        Q = self.query(x)

        K = self.key(x)

        V = self.value(x)

        return Q, K, V
