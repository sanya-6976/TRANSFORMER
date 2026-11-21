import torch

from tasks.model.feed_forward_residual import FeedForwardResidual

torch.manual_seed(42)

batch_size = 2
sequence_length = 5
embedding_dim = 128

x = torch.randn(
    batch_size,
    sequence_length,
    embedding_dim,
    requires_grad=True
)

block = FeedForwardResidual(
    embedding_dim=embedding_dim,
    expansion_factor=4
)

output = block(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)

assert output.shape == x.shape

loss = output.mean()
loss.backward()

assert x.grad is not None

print("Input gradients calculated:", x.grad is not None)
print("All tests passed!")