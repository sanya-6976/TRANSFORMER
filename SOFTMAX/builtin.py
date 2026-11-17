
import torch
import torch.nn.functional as F

x = torch.tensor([2.0, 1.0, 0.1])


def stable_softmax(x):

    shifted = x - torch.max(x)

    exp_values = torch.exp(shifted)

    return exp_values / torch.sum(exp_values)


# Our implementation
manual_result = stable_softmax(x)

# PyTorch implementation
torch_result = F.softmax(x, dim=0)

print("Manual Softmax:", manual_result)
print("PyTorch Softmax:", torch_result)

# Verify both are approximately equal
print("\nAre results equal?",
      torch.allclose(manual_result, torch_result))
