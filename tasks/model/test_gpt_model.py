
import torch

from tasks.model.gpt_model import GPTModel


torch.manual_seed(42)


# Model configuration
batch_size = 2
sequence_length = 5
vocab_size = 10000
max_seq_len = 64
embedding_dim = 128
num_heads = 4
num_layers = 6
expansion_factor = 4


# Create sample token IDs
token_ids = torch.randint(
    0,
    vocab_size,
    (batch_size, sequence_length)
)

# Initialize complete GPT model
model = GPTModel(
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    expansion_factor=expansion_factor,
    dropout=0.0
)


# Forward pass
logits = model(token_ids)


# Test 1: Output shape
print("Input token shape:", token_ids.shape)
print("GPT output shape:", logits.shape)

assert logits.shape == (
    batch_size,
    sequence_length,
    vocab_size
)

print("Output shape test passed!")


# Test 2: Transformer block count
print(
    "Number of Transformer blocks:",
    len(model.transformer_stack.blocks)
)

assert len(model.transformer_stack.blocks) == num_layers

print("Transformer block count test passed!")


# Test 3: Output validity
assert torch.isfinite(logits).all()

print("All logits are finite!")


# Test 4: Trainable parameters
total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("Total trainable parameters:", total_parameters)


# Test 5: Backpropagation
targets = torch.randint(
    0,
    vocab_size,
    (batch_size, sequence_length)
)

loss = torch.nn.functional.cross_entropy(
    logits.reshape(-1, vocab_size),
    targets.reshape(-1)
)

print("Training loss:", loss.item())

loss.backward()

for name, parameter in model.named_parameters():

    assert parameter.grad is not None, (
        f"Missing gradient: {name}"
    )

    assert torch.isfinite(parameter.grad).all(), (
        f"Invalid gradient: {name}"
    )

print("All GPT parameter gradients calculated!")


# Test 6: Maximum sequence length protection
try:
    invalid_tokens = torch.randint(
        0,
        vocab_size,
        (batch_size, max_seq_len + 1)
    )

    model(invalid_tokens)

    raise AssertionError(
        "Model should reject sequences longer than max_seq_len"
    )

except ValueError:
    print("Maximum sequence length test passed!")


print("\nAll complete GPT model tests passed!")
