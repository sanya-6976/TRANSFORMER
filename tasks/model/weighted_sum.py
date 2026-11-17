
import torch
import torch.nn as nn


class WeightedSum(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, attention_weights, V):

        # Combine Value vectors using attention weights
        output = torch.matmul(attention_weights, V)

        return output
