import torch 

from tasks.model.language_model_head import LanguageModelHead


# Reproducibility
torch.manual_seed(42)


# Configuration
batch_size = 2
sequence_length = 5
embedding_dim = 128
vocab_size = 10000

# Create sample Transformer output
x = torch.randn(
    batch_size,
    sequence_length,
    embedding_dim,
    requires_grad=True
)


# Initialize Language Model Head
model = LanguageModelHead(
    embedding_dim=embedding_dim,
    vocab_size=vocab_size
)


# Forward pass
logits = model(x)


# Test 1: Shape verification
print("Input shape:", x.shape)
print("Logits shape:", logits.shape)

assert logits.shape == (
    batch_size,
    sequence_length,
    vocab_size
)

print("Shape test passed!")


# Test 2: Parameter count
total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("Trainable parameters:", total_parameters)

assert total_parameters == (
    embedding_dim * vocab_size + vocab_size
)

print("Parameter count test passed!")


# Test 3: Output validity
assert torch.isfinite(logits).all()

print("All logits are finite!")


# Test 4: Backpropagation
loss = logits.mean()
loss.backward()

assert x.grad is not None

for name, parameter in model.named_parameters():
    assert parameter.grad is not None, f"Missing gradient: {name}"

print("Input gradients calculated:", x.grad is not None)
print("All parameter gradients calculated!")


print("\nAll Language Model Head tests passed!")
