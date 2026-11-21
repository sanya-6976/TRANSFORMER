import torch 
import torch.nn as nn 

class PreNormResidual(nn.Module):
    def __init__(self, embedding_dim, sublayer):
        super().__init__()

        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.sublayer = sublayer

    def forward(self, x):

        normalized = self.layer_norm(x)

        transformed = self.sublayer(normalized)

        output = x + transformed

        return output
   