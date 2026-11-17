
from tasks.tokenizer.bpe_tokenizer import BPETokenizer


# Training corpus
text = "low lower lowest low lower"

# Create BPE tokenizer
tokenizer = BPETokenizer(
    text,
    num_merges=10
)

print("\nFinal Vocabulary:")
print(tokenizer.vocab)

print("\nVocabulary Size:")
print(tokenizer.vocab_size)

# Encode text
sample = "lower"

encoded = tokenizer.encode(sample)

print("\nOriginal Text:")
print(sample)

print("\nBPE Tokens:")
print(encoded)

# Convert tokens into IDs
encoded_ids = tokenizer.encode_ids(sample)

print("\nToken IDs:")
print(encoded_ids)

# Decode
decoded = tokenizer.decode(encoded)

print("\nDecoded Text:")
print(decoded)

# Verify
print("\nRound Trip:")
print(decoded == sample)
