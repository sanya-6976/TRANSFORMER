import torch 
import torch.nn as nn 

class LanguageModelHead(nn.Module):

    def __init__(self, embedding_dim, vocab_size):
        super().__init__()

         # Convert hidden representations into vocabulary scores
        self.projection = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, x):

        # Project hidden representations into vocabulary logits
        logits = self.projection(x)

        return logits