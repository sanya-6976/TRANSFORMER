
import torch


def stable_softmax(x):

    # Step 1: Find maximum value
    max_value = torch.max(x)

    # Step 2: Subtract maximum
    shifted_x = x - max_value

    # Step 3: Exponentiation
    exp_values = torch.exp(shifted_x)

    # Step 4: Normalize
    probabilities = exp_values / torch.sum(exp_values)

    return probabilities


# Normal input
x = torch.tensor([2.0, 1.0, 0.1])

result = stable_softmax(x)

print("Stable Softmax:", result)
print("Sum:", result.sum())


# Test with large values
large_x = torch.tensor([1000.0, 1001.0, 1002.0])

large_result = stable_softmax(large_x)

print("\nLarge input:", large_x)
print("Stable result:", large_result)
print("Probability sum:", large_result.sum())
