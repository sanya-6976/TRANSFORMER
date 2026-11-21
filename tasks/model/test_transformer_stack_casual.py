import torch

from tasks.model.transformer_stack import TransformerStack


# Reproducibility
torch.manual_seed(42)


# Configuration
batch_size = 1
sequence_length = 5
embedding_dim = 128
num_heads = 4
num_layers = 6

# Create Transformer Stack
model = TransformerStack(
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    expansion_factor=4,
    dropout=0.0
)

# Evaluation mode disables dropout
model.eval()

# Create original input
x1 = torch.randn(
    batch_size,
    sequence_length,
    embedding_dim
)

# Create a copy of the input
x2 = x1.clone()

# Change only the final token
x2[:, 4, :] = torch.randn(batch_size, embedding_dim)


# Forward passes
with torch.no_grad():
    output1 = model(x1)
    output2 = model(x2)


# Display shapes
print("Input shape:", x1.shape)
print("Output 1 shape:", output1.shape)
print("Output 2 shape:", output2.shape)


# Compare outputs before the changed token
earlier_tokens_1 = output1[:, :4, :]
earlier_tokens_2 = output2[:, :4, :]

print(
    "Earlier token outputs unchanged:",
    torch.allclose(
        earlier_tokens_1,
        earlier_tokens_2,
        atol=1e-5
    )
)


# Verify causal masking
assert torch.allclose(
    earlier_tokens_1,
    earlier_tokens_2,
    atol=1e-5
), "Causal masking failed!"

print("Causal masking verified across all Transformer blocks!")


# Verify final token can change
final_token_changed = not torch.allclose(
    output1[:, 4, :],
    output2[:, 4, :],
    atol=1e-5
)

print("Final token output changed:", final_token_changed)

assert final_token_changed

print("\nAll causal masking tests passed!")