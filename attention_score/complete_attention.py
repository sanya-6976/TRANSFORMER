
import torch


def scaled_dot_product_attention(Q, K, V):

    # Step 1: Calculate raw attention scores
    scores = K @ Q

    # Step 2: Scale scores
    d_k = K.shape[-1]

    scores = scores / (d_k ** 0.5)

    # Step 3: Stable Softmax
    shifted_scores = scores - torch.max(scores)

    exp_scores = torch.exp(shifted_scores)

    attention_weights = (
        exp_scores / torch.sum(exp_scores)
    )

    # Step 4: Weighted sum of Values
    output = attention_weights @ V

    return output, attention_weights


# Query
Q = torch.tensor(
    [1, 0, 1, 0],
    dtype=torch.float32
)

# Keys
K = torch.tensor([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 1, 0, 0]
], dtype=torch.float32)

# Values
V = torch.tensor([
    [10, 0],
    [0, 10],
    [5, 5]
], dtype=torch.float32)


output, weights = scaled_dot_product_attention(Q, K, V)

print("Attention Weights:")
print(weights)

print("\nFinal Attention Output:")
print(output)

print("\nWeight Sum:")
print(weights.sum())
