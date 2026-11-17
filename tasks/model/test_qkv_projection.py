
import torch

from tasks.model.qkv_projection import QKVProjection


# Configuration
batch_size = 2
sequence_length = 3
embedding_dim = 4


# Create input embeddings
x = torch.randn(
    batch_size,
    sequence_length,
    embedding_dim
)

print("Input Shape:", x.shape)


# Create QKV projection module
qkv = QKVProjection(embedding_dim)


# Generate Q, K and V
Q, K, V = qkv(x)


# Display output shapes
print("\nQuery Shape:", Q.shape)
print("Key Shape:", K.shape)
print("Value Shape:", V.shape)


# Display parameter count
total_parameters = sum(
    p.numel() for p in qkv.parameters()
)

print("\nTotal Parameters:", total_parameters)


# Check output shapes
assert Q.shape == x.shape
assert K.shape == x.shape
assert V.shape == x.shape

print("\nAll shape tests passed!")
