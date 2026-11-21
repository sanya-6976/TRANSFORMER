import torch 
from tasks.model.layer_norm import LayerNorm

torch.manual_seed(42)

# Configuration
batch_size = 2
sequence_length = 5
embedding_dim = 128

# Random input
x = torch.randn(batch_size, sequence_length, embedding_dim)

# Create LayerNorm
model = LayerNorm(embedding_dim)

# Forward pass
output = model(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)

# Check mean and variance for each token
output_mean = output.mean(dim=-1)
output_variance = output.var(dim=-1, unbiased=False)

print("\nMean of normalized features:")
print(output_mean)

print("\nVariance of normalized features:")
print(output_variance)

total_params = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

print("\nTrainable parameters:", total_params)

# Assertions
assert output.shape == x.shape
assert torch.allclose(
    output_mean,
    torch.zeros_like(output_mean), # check teh mean is near zero 
    atol=1e-5
)
assert torch.allclose(
    output_variance,
    torch.ones_like(output_variance),
    atol=1e-4
)

assert total_params == 256

# Backpropagation
loss = output.sum()
loss.backward()

assert model.gamma.grad is not None
assert model.beta.grad is not None

print("\nAll tests passed!")
print("LayerNorm gradients calculated successfully.")


#inbuilt layernorm in pytorch 

pytorch_layer_norm = torch.nn.LayerNorm(embedding_dim)

# Copy identical parameters
with torch.no_grad():
    pytorch_layer_norm.weight.copy_(model.gamma)
    pytorch_layer_norm.bias.copy_(model.beta)

pytorch_output = pytorch_layer_norm(x)

print(
    "\nCustom vs PyTorch:",
    torch.allclose(output, pytorch_output, atol=1e-5)
)