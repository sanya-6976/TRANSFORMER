
import torch

# Input logits
x = torch.tensor([2.0, 1.0, 0.1])

# Step 1: Exponentiation
exp_values = torch.exp(x)

# Step 2: Sum of exponentials
total = torch.sum(exp_values)

# Step 3: Normalize
probabilities = exp_values / total

# Display results
print("Original logits:", x)
print("Exponentials:", exp_values)
print("Sum:", total)

print("\nSoftmax probabilities:")
print(probabilities)

print("\nProbability sum:", probabilities.sum())
