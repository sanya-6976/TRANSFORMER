
import torch

Q = torch.tensor([1, 0, 1, 0], dtype=torch.float32)

K = torch.tensor([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 1, 0, 0]
], dtype=torch.float32)

# Step 1: Calculate raw scores
raw_scores = K @ Q

# Step 2: Find key dimension
d_k = K.shape[-1]

print(d_k)
# Step 3: Calculate scaling factor
scale = torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
print(scale)

# Step 4: Scale scores
scaled_scores = raw_scores / scale

print("Raw scores:", raw_scores)
print("Key dimension:", d_k)
print("Scaling factor:", scale)
print("Scaled scores:", scaled_scores)