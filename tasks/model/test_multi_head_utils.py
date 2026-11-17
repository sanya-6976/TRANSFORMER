import torch 

from tasks.model.multi_head_utils import (
    split_heads,
    combine_heads
)

# Create input
x = torch.randn(
    batch_size,
    sequence_length,
    embedding_dim
)

print("Original Shape:", x.shape)


# Split into heads
split = split_heads(x, num_heads)

print("\nAfter Splitting:", split.shape)


# Combine heads
combined = combine_heads(split)

print("After Combining:", combined.shape)


# Verify dimensions
assert split.shape == (2, 2, 3, 4)

assert combined.shape == (2, 3, 8)


# Verify values are preserved
assert torch.allclose(x, combined)


print("\nAll multi-head shape tests passed!")
