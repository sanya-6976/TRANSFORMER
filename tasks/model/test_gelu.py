import torch 
import torch.nn as nn 

# Sample input
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# ReLU activation
relu = nn.ReLU()
relu_output = relu(x)

# GELU activation
gelu = nn.GELU()
gelu_output = gelu(x)

print("Input:       ", x)
print("ReLU output: ", relu_output)
print("GELU output: ", gelu_output)
