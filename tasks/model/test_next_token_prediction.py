import torch

from tasks.model.next_token_prediction import predict_next_token

# Three positions, five vocabulary tokens
logits = torch.tensor([
    [
        [1.0, 2.0, 0.5, 0.2, 0.1],
        [0.5, 0.3, 2.5, 1.0, 0.2],
        [0.1, 0.4, 1.5, 3.2, 0.8]
    ]
])

print("Logits shape:", logits.shape)

next_token = predict_next_token(logits)

print("Next token:", next_token)
print("Next token shape:", next_token.shape)

assert next_token.shape == (1,)
assert next_token.item() == 3

print("Next-token prediction test passed!")
