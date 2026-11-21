import torch

from tasks.model.input_embedding import InputEmbedding
from tasks.model.transformer_stack import TransformerStack


torch.manual_seed(42)


# Configuration
batch_size = 2
sequence_length = 5
vocab_size = 100
max_seq_len = 64
embedding_dim = 128
num_heads = 4
num_layers = 6

# create sample token ids 
token_ids = torch.randint(
    0 , 
    vocab_size , 
    (batch_size , sequence_length)
)

# Step 1: Input Embedding
input_embedding = InputEmbedding(
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
    embedding_dim=embedding_dim
)

# Step 2: Transformer Stack
transformer_stack = TransformerStack(
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    expansion_factor=4,
    dropout=0.0
)
# Step 3: Combine the components
input_vectors = input_embedding(token_ids)

output = transformer_stack(input_vectors)

# Step 4: Verify shapes
print("Token IDs shape:", token_ids.shape)
print("Input embedding shape:", input_vectors.shape)
print("Transformer output shape:", output.shape)

assert input_vectors.shape == (
    batch_size,
    sequence_length,
    embedding_dim
)

assert output.shape == (
    batch_size,
    sequence_length,
    embedding_dim
)

print("Shape verification passed!")

# Step 5: Verify finite outputs
assert torch.isfinite(output).all()

print("Output contains only finite values!")

# Step 6: Count total parameters
total_parameters = (
    sum(p.numel() for p in input_embedding.parameters())
    + sum(p.numel() for p in transformer_stack.parameters())
)

print("Total trainable parameters:", total_parameters)


# Step 7: Backpropagation
loss = output.mean()
loss.backward()

for name, parameter in input_embedding.named_parameters():
    assert parameter.grad is not None, f"Missing gradient: {name}"

for name, parameter in transformer_stack.named_parameters():
    assert parameter.grad is not None, f"Missing gradient: {name}"

print("Input embedding gradients verified!")
print("Transformer stack gradients verified!")

print("\nComplete Transformer integration successful!")
