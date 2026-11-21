
import torch

from tasks.model.transformer_stack import TransformerStack


# Reproducibility
torch.manual_seed(42)


# Configuration
batch_size = 2
sequence_length = 5
embedding_dim = 128
num_heads = 4
num_layers = 6
expansion_factor = 4


# Create random input
x = torch.randn(
    batch_size,
    sequence_length,
    embedding_dim,
    requires_grad=True
)


# Initialize Transformer Stack
model = TransformerStack(
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    expansion_factor=expansion_factor,
    dropout=0.0
)


# Forward pass
output = model(x)


# Test 1: Shape verification
print("Input shape:", x.shape)
print("Output shape:", output.shape)

assert output.shape == x.shape
print("Shape test passed!")


# Test 2: Number of Transformer blocks
print("Number of Transformer blocks:", len(model.blocks))

assert len(model.blocks) == num_layers
print("Transformer block count test passed!")


# Test 3: Final LayerNorm
print("Final LayerNorm:", model.final_layer_norm)

final_ln_parameters = sum(
    parameter.numel()
    for parameter in model.final_layer_norm.parameters()
)

print("Final LayerNorm parameters:", final_ln_parameters)

assert final_ln_parameters == 2 * embedding_dim
print("Final LayerNorm test passed!")


# Test 4: Total trainable parameters
total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("Total trainable parameters:", total_parameters)


# Test 5: Forward output validity
assert torch.isfinite(output).all()
print("Output contains only finite values!")


# Test 6: Backpropagation
loss = output.mean()
loss.backward()

assert x.grad is not None
print("Input gradients calculated:", x.grad is not None)


# Test 7: Parameter gradients
for name, parameter in model.named_parameters():

    assert parameter.grad is not None, (
        f"Missing gradient: {name}"
    )

    assert torch.isfinite(parameter.grad).all(), (
        f"Invalid gradient: {name}"
    )

print("All parameter gradients calculated!")


print("\nAll Transformer Stack tests passed!")
