import torch 
import torch.nn as nn 

class ResidualConnection(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x , transformed_x):
        output = x + transformed_x
        return output

