import torch

from tasks.model.language_model_loss import LanguageModelLoss

torch.manual_seed(42)

batch_size = 2
sequence_length = 4
vocab_size = 1000

# Simulated GPT logits
logits = torch.randn(
    batch_size,
    sequence_length,
    vocab_size,
    requires_grad=True
)

# Correct next-token IDs
targets = torch.randint(
    0,
    vocab_size,
    (batch_size, sequence_length)
)

# Initialize loss function
loss_function = LanguageModelLoss()

# Calculate loss
loss = loss_function(logits, targets)

print("Logits shape:", logits.shape)
print("Targets shape:", targets.shape)
print("Loss:", loss.item())

# Verify scalar loss
assert loss.ndim == 0

# Verify finite loss
assert torch.isfinite(loss)

print("Loss calculation passed!")

# Backpropagation
loss.backward()

assert logits.grad is not None

print("Gradients calculated:", logits.grad is not None)

print("\nAll language model loss tests passed!")
