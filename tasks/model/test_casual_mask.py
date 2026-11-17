
import torch

from tasks.model.casual_mask import CausalMask


# Example attention scores
scores = torch.tensor([
    [2.0, 1.0, 3.0],
    [1.0, 2.0, 4.0],
    [3.0, 1.0, 2.0]
])


# Create causal mask
causal_mask = CausalMask()

# Apply mask
masked_scores = causal_mask(scores)


# Apply Softmax
attention_weights = torch.softmax(
    masked_scores,
    dim=-1
)


print("Original Scores:")
print(scores)

print("\nMasked Scores:")
print(masked_scores)

print("\nAttention Probabilities:")
print(attention_weights)


# Verify future positions have zero probability
assert torch.allclose(
    attention_weights[0, 1:],
    torch.zeros(2)
)

assert attention_weights[1, 2].item() == 0

# Verify every row sums to one
assert torch.allclose(
    attention_weights.sum(dim=-1),
    torch.ones(3)
)

print("\nAll causal masking tests passed!")
