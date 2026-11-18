import torch 
import torch.nn as nn 

class FeedForward(nn.Module):
     def __init__(self, embedding_dim, expansion_factor=4, dropout=0.0):
        super().__init__()
        hidden_dim = embedding_dim * expansion_factor #hidden_dim = embedding_dim * expansion_factor

        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout)
        )

     def  forward(self , x):
        output = self.network(x)
        return output
    