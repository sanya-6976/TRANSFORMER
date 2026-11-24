
import torch

from tasks.model.gpt_model import GPTModel


torch.manual_seed(42)

# Configuration
vocab_size = 10000
max_seq_len = 64
embedding_dim = 128
num_heads = 4
num_layers = 6
batch_size = 2
sequence_length = 5


# Create GPT with weight tying enabled
model = GPTModel(
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    expansion_factor=4,
    dropout=0.0,
    tie_weights=True
)


# Access the two weight parameters
input_weight = (
    model.input_embedding.token_embedding.embedding.weight
)

output_weight = (
    model.language_model_head.projection.weight
)


# Test 1: Same parameter object
print(
    "Same parameter object:",
    input_weight is output_weight
)

assert input_weight is output_weight

print("Weight sharing test passed!")


# Test 2: Parameter count
total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("Total trainable parameters:", total_parameters)

assert total_parameters == 2488080

print("Parameter reduction test passed!")


# Test 3: Forward pass
token_ids = torch.randint(
    0,
    vocab_size,
    (batch_size, sequence_length)
)

logits = model(token_ids)

print("Output shape:", logits.shape)

assert logits.shape == (
    batch_size,
    sequence_length,
    vocab_size
)

print("Forward pass test passed!")


# Test 4: Backpropagation
targets = torch.randint(
    0,
    vocab_size,
    (batch_size, sequence_length)
)

loss = torch.nn.functional.cross_entropy(
    logits.reshape(-1, vocab_size),
    targets.reshape(-1)
)

loss.backward()

assert input_weight.grad is not None
assert torch.isfinite(input_weight.grad).all()

print("Shared embedding gradients calculated!")


print("\nAll Weight Tying tests passed!")
