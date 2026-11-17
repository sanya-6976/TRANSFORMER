
import torch

from tasks.model.attention_score import AttentionScores


# Manually defined Query matrix
Q = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

# Manually defined Key matrix
K = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])


# Create attention score module
attention = AttentionScores()

# Calculate scores
scores = attention(Q, K)


print("Query Shape:", Q.shape)
print("Key Shape:", K.shape)

print("\nAttention Scores:")
print(scores)

print("\nScore Matrix Shape:", scores.shape)


# Verify expected result
expected = torch.tensor([
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 1.0, 2.0]
])

assert torch.allclose(scores, expected)

print("\nAttention score test passed!")
