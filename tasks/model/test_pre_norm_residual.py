import torch 
import torch.nn as nn 

from tasks.model.pre_norm_residual import PreNormResidual

torch.manual_seed(42)

# Configuration
batch_size = 2
sequence_length = 5
embedding_dim = 128

# Input
x = torch.randn(
    batch_size,
    sequence_length,
    embedding_dim,
    requires_grad=True
)

# Feed-forward sublayer
feed_forward = nn.Sequential(
    nn.Linear(embedding_dim , embedding_dim * 4),
    nn.GELU(),
    nn.Linear(embedding_dim * 4, embedding_dim)
)

# pre-LN residual module 
block = PreNormResidual(
    embedding_dim , feed_forward
)

# Forward pass
output = block(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)

# Check shape preservation
assert output.shape == x.shape

# Backpropagation
loss = output.mean()
loss.backward()

assert x.grad is not None

for parameter in block.parameters():
    assert parameter.grad is not None

print("Input gradients calculated:", x.grad is not None)

print("All tests passed!")


