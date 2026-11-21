import torch 
from tasks.model.language_model_head import LanguageModelHead 

torch.manual_seed(42)

batch_size = 2
sequence_length = 5
embedding_dim = 128
vocab_size = 10000

# Simulate Transformer output
x = torch.randn(
    batch_size,
    sequence_length,
    embedding_dim
)

# Generate logits
model = LanguageModelHead(
    embedding_dim=embedding_dim,
    vocab_size=vocab_size
)

logits = model(x)

# Convert logits into probabilities
probabilities = torch.softmax(logits, dim=-1)

print("Logits shape:", logits.shape)
print("Probabilities shape:", probabilities.shape)

# Check shape
assert probabilities.shape == logits.shape

# Check probability range
assert torch.all(probabilities >= 0)
assert torch.all(probabilities <= 1)

# Check that probabilities sum to 1
probability_sums = probabilities.sum(dim=-1)

print("Probability sums:", probability_sums)

assert torch.allclose(
    probability_sums,
    torch.ones_like(probability_sums),
    atol=1e-5
)

print("All probabilities are valid!")
print("Softmax tests passed!")
