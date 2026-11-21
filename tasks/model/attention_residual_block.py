import torch
import torch.nn as nn

from tasks.model.multi_head_casual_attention import MultiHeadCausalAttention

class AttentionResidualBlock(nn.Module):

    def __init__(self, embedding_dim, num_heads, dropout=0.0):
        super().__init__()

        self.layer_norm = nn.LayerNorm(embedding_dim)

        self.attention = MultiHeadCausalAttention(
            embedding_dim = embedding_dim , 
            num_heads = num_heads , 
            dropout = dropout
        )

    def forward(self, x):

        normalized = self.layer_norm(x)

        attention_output, _ = self.attention(normalized)

        output = x + attention_output

        return output
