
import torch

from tasks.model.input_embedding import InputEmbedding


# Configuration
vocab_size = 100
max_seq_len = 64
embedding_dim = 16

# Create complete input embedding layer
input_layer = InputEmbedding(
    vocab_size,
    max_seq_len,
    embedding_dim
)

# Two sequences, four tokens each
token_ids = torch.tensor(
    [
        [4, 8, 12, 15],
        [3, 7, 9, 11]
    ],
    dtype=torch.long
)

print("Input Token IDs:")
print(token_ids)

print("\nInput Shape:")
print(token_ids.shape)

# Pass through complete embedding layer
output = input_layer(token_ids)

print("\nFinal Embedding Output:")
print(output)

print("\nFinal Output Shape:")
print(output.shape)

# Count parameters
total_parameters = sum(
    p.numel()
    for p in input_layer.parameters()
    if p.requires_grad
)

print("\nTotal Trainable Parameters:")
print(total_parameters)
