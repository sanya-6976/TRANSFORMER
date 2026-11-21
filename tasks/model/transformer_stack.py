import torch
import torch.nn as nn

from tasks.model.transformer_block import TransformerBlock


class TransformerStack(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        num_layers,
        expansion_factor=4,
        dropout=0.0
    ):
         super().__init__()

         self.blocks = nn.ModuleList([
              TransformerBlock(
                   embedding_dim = embedding_dim , 
                   num_heads=num_heads,
                expansion_factor=expansion_factor,
                dropout=dropout
              )
              for _ in range(num_layers)
         ])

          # Final Layer Normalization
         self.final_layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):

        # Pass input through every Transformer block
        for block in self.blocks:
            x = block(x)

        # Normalize the final output
        x = self.final_layer_norm(x)

        return x

    