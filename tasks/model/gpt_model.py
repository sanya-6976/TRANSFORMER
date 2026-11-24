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
        dropout=0.0
    ):
        super().__init__()

        # step 1 :- covert token iDS INTO TOKEN + POSITIONAL EMBEDDING S
        self.input_embedding = InputEmbedding(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            embedding_dim=embedding_dim
        )

        #step 2 :-Process embeddings through transfromer blocks 
        self.transformer_block = TransformerBlock

