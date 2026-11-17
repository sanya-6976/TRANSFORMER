
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):

    def __init__(self, max_seq_len, embedding_dim):

        super().__init__()

        # Create a learnable embedding for every position
        self.embedding = nn.Embedding(
            num_embeddings=max_seq_len,
            embedding_dim=embedding_dim
        )

    def forward(self, sequence_length):

        # Generate position IDs: 0, 1, 2, ... T-1
        positions = torch.arange(sequence_length)

        # Retrieve positional vectors
        position_embeddings = self.embedding(positions)

        return position_embeddings
