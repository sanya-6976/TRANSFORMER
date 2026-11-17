
import torch
import torch.nn as nn


class AttentionSoftmax(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, scores):

        # Convert raw attention scores into probabilities
        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        return attention_weights
