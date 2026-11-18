import torch 

from tasks.model.multi_head_casual_attention import (
    MultiHeadCausalAttention
)

# Reproducible random values
torch.manual_seed(42)

# Configuration
batch_size = 2
sequence_length = 3
embedding_dim = 8
num_heads = 2

# Random input
x = torch.randn(
    batch_size,
    sequence_length,
    embedding_dim
)

# Create attention layer
model = MultiHeadCausalAttention(
    embedding_dim=embedding_dim,
    num_heads=num_heads
)


# Forward pass
output, attention_weights = model(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Attention shape:", attention_weights.shape)

print("\nAttention weights:")
print(attention_weights)

# Check output shape
assert output.shape == (2, 3, 8)

# Check attention shape
assert attention_weights.shape == (2, 2, 3, 3)

# Check each row sums to 1
row_sums = attention_weights.sum(dim=-1)
assert torch.allclose(
    row_sums,
    torch.ones_like(row_sums),
    atol=1e-6
)
# Check future positions have zero attention
assert torch.all(attention_weights.triu(1) == 0)

# Check gradients
loss = output.sum()
loss.backward()

assert model.qkv_projection.weight.grad is not None
assert model.output_projection.weight.grad is not None

print("\nAll tests passed!")
print("Gradients successfully calculated.")
