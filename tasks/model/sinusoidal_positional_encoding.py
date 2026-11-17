
import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):

    def __init__(self, max_seq_len, embedding_dim):
        super().__init__()

        # Create a matrix for all positions and dimensions
        pe = torch.zeros(max_seq_len, embedding_dim)

        # Generate all position indices
        positions = torch.arange(max_seq_len).unsqueeze(1)

        # Calculate the frequency for each dimension pair
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2)
            * (-math.log(10000.0) / embedding_dim)
        )

        # Apply sine to even dimensions
        pe[:, 0::2] = torch.sin(positions * div_term)

        # Apply cosine to odd dimensions
        pe[:, 1::2] = torch.cos(positions * div_term)

        # Store encoding without making it trainable
        self.register_buffer("pe", pe)

    def forward(self, sequence_length):

        return self.pe[:sequence_length]
