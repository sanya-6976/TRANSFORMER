
import torch

from tasks.model.positional_embedding import PositionalEmbedding


# Configuration
max_seq_len = 64
embedding_dim = 128

# Create positional embedding layer
position_layer = PositionalEmbedding(
    max_seq_len,
    embedding_dim
)

# Suppose our input has 4 tokens
sequence_length = 4

# Get positional embeddings
position_embeddings = position_layer(sequence_length)

print("Position IDs:")
print(torch.arange(sequence_length))

print("\nPositional Embeddings:")
print(position_embeddings)

print("\nOutput Shape:")
print(position_embeddings.shape)

# Count trainable parameters
parameters = sum(
    p.numel()
    for p in position_layer.parameters()
    if p.requires_grad
)

print("\nTrainable Parameters:")
print(parameters)
