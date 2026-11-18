import torch

from tasks.model.feed_forward import FeedForward

# Reproducible results
torch.manual_seed(42)

# Configuration
batch_size = 2
sequence_length = 5
embedding_dim = 128

# Random input
x = torch.randn(batch_size, sequence_length, embedding_dim)

# Create FFN
model = FeedForward(
    embedding_dim=embedding_dim,
    expansion_factor=4,
    dropout=0.0
)

# Forward pass
output = model(x)

# Display shapes
print("Input shape:", x.shape)
print("Output shape:", output.shape)

# Count trainable parameters
total_params = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("Trainable parameters:", total_params)

# Shape verification
assert output.shape == (2, 5, 128)

# Backpropagation test
loss = output.sum()
loss.backward()

# Verify gradients
for name, parameter in model.named_parameters():
    assert parameter.grad is not None, f"Missing gradient: {name}"

print("\nAll tests passed!")
print("Gradients successfully calculated.")
