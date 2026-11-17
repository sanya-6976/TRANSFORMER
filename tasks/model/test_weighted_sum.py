
import torch

from tasks.model.weighted_sum import WeightedSum


# Attention probability matrix
attention_weights = torch.tensor([
    [0.7, 0.2, 0.1],
    [0.1, 0.6, 0.3],
    [0.2, 0.3, 0.5]
])

# Value matrix
V = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0]
])


# Create weighted sum module
weighted_sum = WeightedSum()

# Calculate output
output = weighted_sum(attention_weights, V)


print("Attention Weights:")
print(attention_weights)

print("\nValue Matrix:")
print(V)

print("\nWeighted Sum Output:")
print(output)

print("\nOutput Shape:", output.shape)


# Verify expected output
expected = torch.tensor([
    [1.8, 2.8],
    [3.4, 4.4],
    [3.6, 4.6]
])

assert torch.allclose(output, expected)

print("\nWeighted sum test passed!")
