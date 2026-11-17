
import torch

# Query vector
Q = torch.tensor(
    [1, 0, 1, 0],
    dtype=torch.float32
)

# Three Key vectors
K = torch.tensor([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 1, 0, 0]
], dtype=torch.float32)

# Calculate raw dot-product scores
raw_scores = K @ Q

print("Query:", Q)
print("\nKeys:\n", K)

print("\nRaw Attention Scores:")
print(raw_scores)
