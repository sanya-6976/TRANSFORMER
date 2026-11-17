
import torch
import torch.nn as nn

from tasks.model.embedding import TokenEmbedding
from tasks.model.positional_embedding import PositionalEmbedding


class InputEmbedding(nn.Module):

    def __init__(self, vocab_size, max_seq_len, embedding_dim):

        super().__init__()

        # Token embedding layer
        self.token_embedding = TokenEmbedding(
            vocab_size,
            embedding_dim
        )

        # Positional embedding layer
        self.position_embedding = PositionalEmbedding(
            max_seq_len,
            embedding_dim
        )

    def forward(self, token_ids):

        # Shape: (B, T, C)
        token_vectors = self.token_embedding(token_ids)

        # Get sequence length
        sequence_length = token_ids.shape[1]

        # Shape: (T, C)
        position_vectors = self.position_embedding(sequence_length)

        # Add positional information to token information
        output = token_vectors + position_vectors.unsqueeze(0)

        return output
