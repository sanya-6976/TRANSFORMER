
import torch

from tasks.model.transformer_block import TransformerBlock

torch.manual_seed(42)

# Configuration
batch_size = 2
sequence_length = 5
embedding_dim = 128
num_heads = 4

# Input tensor
x = torch.randn(
    batch_size,
    sequence_length,
    embedding_dim,
    requires_grad=True
)

# Create Transformer Block
model = TransformerBlock(
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    expansion_factor=4,
    dropout=0.0
)

# Forward pass
output = model(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)

# Shape verification
assert output.shape == x.shape

# Parameter count
total_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("Trainable parameters:", total_parameters)

# Backpropagation
loss = output.mean()
loss.backward()

assert x.grad is not None

for name, parameter in model.named_parameters():
    assert parameter.grad is not None, f"Missing gradient: {name}"

print("Input gradients calculated:", x.grad is not None)
print("All parameter gradients calculated!")

print("All tests passed!")
