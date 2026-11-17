
from tasks.tokenizer.character_tokenizer import CharacterTokenizer

# Step 1: Training corpus
text = "hello world\nthis is our tiny gpt transformer"

# Step 2: Create tokenizer object
tokenizer = CharacterTokenizer(text)

# Step 3: Prepare sentences
sentences = [
    "hello",
    "world",
    "transformer"
]

# Step 4: Encode all sentences
sequences = [
    tokenizer.encode(sentence)
    for sentence in sentences
]

print("Original sequences:")
for sequence in sequences:
    print(sequence)

# Step 5: Pad sequences
padded = tokenizer.pad_sequences(sequences)

print("\nPadded sequences:")
for sequence in padded:
    print(sequence)

# Step 6: Truncate a sequence
truncated = tokenizer.truncate(
    sequences[2],
    max_length=5
)

print("\nTruncated sequence:")
print(truncated)

# Step 7: Convert to PyTorch tensor
import torch

input_ids = torch.tensor(
    padded,
    dtype=torch.long
)

print("\nPyTorch Tensor:")
print(input_ids)

print("\nTensor Shape:")
print(input_ids.shape)
