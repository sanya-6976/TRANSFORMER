
import torch
import torch.nn as nn


class LanguageModelLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, logits, targets):

        # Get dimensions
        batch_size, sequence_length, vocab_size = logits.shape

        # Flatten logits: (B, T, V) -> (B*T, V)
        logits = logits.reshape(
            batch_size * sequence_length,
            vocab_size
        )

        # Flatten targets: (B, T) -> (B*T)
        targets = targets.reshape(
            batch_size * sequence_length
        )

        # Calculate cross-entropy loss
        loss = self.loss_function(logits, targets)

        return loss

