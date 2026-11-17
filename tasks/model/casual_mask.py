
import torch
import torch.nn as nn


class CausalMask(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, scores):

        # Get the sequence length
        sequence_length = scores.shape[-1]

        # Create a lower-triangular mask
        mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=scores.device,
                dtype=torch.bool
            )
        )

        # Replace future-token scores with negative infinity
        masked_scores = scores.masked_fill(
            ~mask,
            float("-inf")
        )

        return masked_scores
