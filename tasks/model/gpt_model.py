
import torch
import torch.nn as nn

from tasks.model.input_embedding import InputEmbedding
from tasks.model.transformer_stack import TransformerStack
from tasks.model.language_model_head import LanguageModelHead


class GPTModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        max_seq_len,
        embedding_dim,
        num_heads,
        num_layers,
        expansion_factor=4,
        dropout=0.0, 
        tie_weights = True
    ):
        super().__init__()

        # 1. Convert token IDs into token + position embeddings
        self.input_embedding = InputEmbedding(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            embedding_dim=embedding_dim
        )

        # 2. Process embeddings through Transformer blocks
        self.transformer_stack = TransformerStack(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            expansion_factor=expansion_factor,
            dropout=dropout
        )

        # 3. Convert final hidden representations into vocabulary logits
        self.language_model_head = LanguageModelHead(
            embedding_dim=embedding_dim,
            vocab_size=vocab_size
        )

        # step 4:- optionally share input an dthe output weights 
        if tie_weights: 
            self.language_model_head.projection.weight = (
                self.input_embedding.token_embedding.embedding.weight
            )

        self.tie_weights = tie_weights
        self.max_seq_len = max_seq_len

    def forward(self, token_ids):

        # Check that input contains batch and sequence dimensions
        if token_ids.ndim != 2:
            raise ValueError(
                "token_ids must have shape (batch_size, sequence_length)"
            )

        # Check maximum supported sequence length
        if token_ids.shape[1] > self.max_seq_len:
            raise ValueError(
                f"Sequence length cannot exceed {self.max_seq_len}"
            )

        # Step 1: Token and positional embeddings
        x = self.input_embedding(token_ids)

        # Step 2: Transformer processing
        x = self.transformer_stack(x)

        # Step 3: Vocabulary logits
        logits = self.language_model_head(x)

        return logits
