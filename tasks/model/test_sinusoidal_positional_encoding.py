
import torch

from tasks.model.sinusoidal_positional_encoding import (
    SinusoidalPositionalEncoding
)

max_seq_len = 10
embedding_dim = 4
sequence_length = 3

positional_encoding = SinusoidalPositionalEncoding(
    max_seq_len,
    embedding_dim
)

output = positional_encoding(sequence_length)

print("Positional Encoding:")
print(output)

print("\nOutput Shape:", output.shape)

print(
    "Trainable Parameters:",
    sum(p.numel() for p in positional_encoding.parameters()
        if p.requires_grad)
)

print("\nPosition 0:", output[0])
print("Position 1:", output[1])
print("Position 2:", output[2])
