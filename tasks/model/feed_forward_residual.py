import torch
import torch.nn as nn

from tasks.model.feed_forward import FeedForward

class FeedForwardResidual(nn.Module):
     def __init__(self, embedding_dim, expansion_factor=4, dropout=0.0):
        super().__init__()

        self.layer_norm = nn.LayerNorm(embedding_dim)

        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim,
            expansion_factor=expansion_factor,
            dropout=dropout
        )

     def forward(self, x):

        normalized = self.layer_norm(x)

        transformed = self.feed_forward(normalized)

        output = x + transformed

        return output
