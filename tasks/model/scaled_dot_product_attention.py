import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V):
        # Get the dimension of each Key vector
        d_k = K.shape[-1]

        # Calculate raw attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # Scale the attention scores
        scaled_scores = scores / math.sqrt(d_k)

        # Convert scores into probabilities
        attention_weights = torch.softmax(
            scaled_scores,
            dim=-1
        )

        # Calculate weighted sum of Values
        output = torch.matmul(attention_weights, V)

        return output, attention_weights


