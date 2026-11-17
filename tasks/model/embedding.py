
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim):

        super().__init__()

        # Create the embedding lookup table
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

    def forward(self, token_ids):

        # Retrieve embedding vectors for token IDs
        embeddings = self.embedding(token_ids)

        return embeddings
