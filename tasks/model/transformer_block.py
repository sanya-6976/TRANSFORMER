import torch
import torch.nn as nn

from tasks.model.multi_head_casual_attention import MultiHeadCausalAttention
from tasks.model.feed_forward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self , embedding_dim ,num_heads ,expansion_factor=4,dropout= 0.0):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(embedding_dim)

        self.attention = MultiHeadCausalAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout
        )

         # Second LayerNorm
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        # Feed-Forward Network
        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim,
            expansion_factor=expansion_factor,
            dropout=dropout
        )

    def forward(self , x):
        normalized = self.layer_norm1(x) 

        attention_output , _ = self.attention(normalized)

        x = x + attention_output

          # Second sublayer: FFN + Residual
        normalized = self.layer_norm2(x)

        feed_forward_output = self.feed_forward(normalized)

        x = x + feed_forward_output

        return x 
