import torch 

from tasks.model.residual_connection import ResidualConnection

torch.manual_seed(42)

# Original input
x = torch.tensor([[2.0, 4.0, 6.0]], requires_grad=True)

# Output from a transformation
transformed_x = torch.tensor([[3.0, 2.0, 1.0]], requires_grad=True)

# Create residual module
residual = ResidualConnection()

# Forward pass
output = residual(x, transformed_x)

print("Original input:", x)
print("Transformed input:", transformed_x)
print("Residual output:", output)

# Verify result
assert torch.equal(output, torch.tensor([[5.0, 6.0, 7.0]]))

# Backpropagation
loss = output.sum()
loss.backward()

print("\nInput gradients:", x.grad)
print("Transformation gradients:", transformed_x.grad)

assert torch.all(x.grad == 1)
assert torch.all(transformed_x.grad == 1)

print("\nAll tests passed!")
