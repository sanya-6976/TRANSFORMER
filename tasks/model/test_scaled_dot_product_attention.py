
import torch

from tasks.model.scaled_dot_product_attention import (
    ScaledDotProductAttention
)


# Define Query
Q = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0]
])

# Define Key
K = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0]
])

# Define Value
V = torch.tensor([
    [10.0, 20.0],
    [30.0, 40.0]
])


# Create attention module
attention = ScaledDotProductAttention()

# Calculate scaled attention
output, weights = attention(Q, K, V)


print("Query Shape:", Q.shape)
print("Key Shape:", K.shape)
print("Value Shape:", V.shape)

print("\nAttention Weights:")
print(weights)

print("\nAttention Output:")
print(output)

print("\nOutput Shape:", output.shape)

# Verify expected dimensions
assert output.shape == V.shape
assert weights.shape == (2, 2)

# Verify probabilities sum to one
assert torch.allclose(
    weights.sum(dim=-1),
    torch.ones(2)
)

print("\nAll scaled attention tests passed!")
