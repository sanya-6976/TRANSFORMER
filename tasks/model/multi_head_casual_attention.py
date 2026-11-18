
import torch
import torch.nn as nn
import math


class MultiHeadCausalAttention(nn.Module):

    def __init__(self, embedding_dim, num_heads, dropout=0.0):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads"
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.qkv_projection = nn.Linear(
            embedding_dim,
            3 * embedding_dim
        )

        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.attention_dropout = nn.Dropout(dropout)

    def forward(self, x):

        batch_size, sequence_length, embedding_dim = x.shape

        # 1. Generate Q, K, V
        qkv = self.qkv_projection(x)
        Q, K, V = qkv.chunk(3, dim=-1)

        # 2. Split into heads
        Q = Q.reshape(
            batch_size, sequence_length,
            self.num_heads, self.head_dim
        ).transpose(1, 2)

        K = K.reshape(
            batch_size, sequence_length,
            self.num_heads, self.head_dim
        ).transpose(1, 2)

        V = V.reshape(
            batch_size, sequence_length,
            self.num_heads, self.head_dim
        ).transpose(1, 2)

        # 3. Calculate attention scores
        scores = torch.matmul(
            Q, K.transpose(-2, -1)
        )

        # 4. Scale scores
        scores = scores / math.sqrt(self.head_dim)

        # 5. Apply causal mask
        mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device,
                dtype=torch.bool
            )
        )

        scores = scores.masked_fill(
            ~mask,
            float("-inf")
        )

        # 6. Calculate attention probabilities
        attention_weights = torch.softmax(
            scores, dim=-1
        )

        dropped_weights = self.attention_dropout(
            attention_weights
        )

        # 7. Weighted sum of values
        output = torch.matmul(dropped_weights, V)

        # 8. Combine heads
        output = output.transpose(1, 2).contiguous()

        output = output.reshape(
            batch_size,
            sequence_length,
            self.embedding_dim
        )

        # 9. Output projection
        output = self.output_projection(output)

        return output, attention_weights
