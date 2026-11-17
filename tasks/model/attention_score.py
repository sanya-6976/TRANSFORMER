
import torch
import torch.nn as nn


class AttentionScores(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, Q, K):

        # Transpose the last two dimensions of K
        K_transposed = K.transpose(-2, -1)

        # Calculate attention scores
        scores = torch.matmul(Q, K_transposed)

        return scores
