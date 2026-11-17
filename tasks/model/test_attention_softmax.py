
import torch

from tasks.model.attention_softmax import AttentionSoftmax


# Raw attention scores
scores = torch.tensor([
    [2.0, 1.0, 0.0],
    [1.0, 3.0, 2.0],
    [0.0, 2.0, 4.0]
])


# Create Softmax module
softmax = AttentionSoftmax()


# Calculate attention probabilities
weights = softmax(scores)


print("Raw Attention Scores:")
print(scores)

print("\nAttention Probabilities:")
print(weights)

print("\nRow Sums:")
print(weights.sum(dim=-1))


# Verify probabilities
assert torch.all(weights >= 0)
assert torch.all(weights <= 1)

assert torch.allclose(
    weights.sum(dim=-1),
    torch.ones(3)
)

print("\nAll Softmax tests passed!")
