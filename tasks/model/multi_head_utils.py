import torch 

def split_heads(x , num_heads):
    batch_size, sequence_length, embedding_dim = x.shape

    head_dim = embedding_dim // num_heads

    x = x.reshape(
        batch_size,
        sequence_length,
        num_heads,
        head_dim
    )

    # Move heads before sequence dimension
    x = x.transpose(1, 2)

    # Output: (B, H, T, head_dim)
    return x

def combine_heads(x):
    # Input: (B, H, T, head_dim)
    batch_size, num_heads, sequence_length, head_dim = x.shape

    # Move heads after sequence dimension
    x = x.transpose(1, 2)

    # Combine heads
    x = x.reshape(
        batch_size,
        sequence_length,
        num_heads * head_dim
    )

    # Output: (B, T, embedding_dim)
    return x