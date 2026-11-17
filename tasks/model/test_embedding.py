
import torch

from tasks.model.embedding import TokenEmbedding


# Model configuration
vocab_size = 100
embedding_dim = 16

# Create embedding layer
embedding_layer = TokenEmbedding(
    vocab_size,
    embedding_dim
)

# Sample token IDs
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

# Pass token IDs through embedding layer
embeddings = embedding_layer(token_ids)

print("\nEmbedding Output:")
print(embeddings)

print("\nEmbedding Shape:")
print(embeddings.shape)

# Count trainable parameters
total_parameters = sum(
    p.numel()
    for p in embedding_layer.parameters()
    if p.requires_grad
)

print("\nTrainable Parameters:")
print(total_parameters)
