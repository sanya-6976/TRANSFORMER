
from collections import Counter


class BPETokenizer:

    def __init__(self, text, num_merges=10):

        self.text = text
        self.num_merges = num_merges

        # Store learned merge operations
        self.merges = []

        # Train BPE
        self.train()

    def merge_pair(self, tokens, pair):

        merged = []
        i = 0

        while i < len(tokens):

            # Check whether adjacent tokens match
            # the pair we want to merge
            if (
                i < len(tokens) - 1
                and (tokens[i], tokens[i + 1]) == pair
            ):

                merged.append(tokens[i] + tokens[i + 1])
                i += 2

            else:
                merged.append(tokens[i])
                i += 1

        return merged

    def train(self):

        # Initially, every character is a token
        tokens = list(self.text)

        for step in range(self.num_merges):

            # Count adjacent pairs
            pair_counts = Counter(
                zip(tokens, tokens[1:])
            )

            # Stop if no pairs remain
            if not pair_counts:
                break

            # Find most frequent pair
            best_pair, frequency = pair_counts.most_common(1)[0]

            # Save the merge rule
            self.merges.append(best_pair)

            # Apply the merge
            tokens = self.merge_pair(tokens, best_pair)

            print(
                f"Merge {step + 1}: "
                f"{best_pair} -> "
                f"{best_pair[0] + best_pair[1]} "
                f"(frequency: {frequency})"
            )

        # Build vocabulary from original characters
        # and all learned merged tokens
        vocabulary = set(self.text)

        for left, right in self.merges:
            vocabulary.add(left + right)

        self.vocab = sorted(vocabulary)

        # Reserve ID 0 for unknown tokens
        self.token_to_id = {"<UNK>": 0}

        for token in self.vocab:
            self.token_to_id[token] = len(self.token_to_id)

        self.id_to_token = {
            index: token
            for token, index in self.token_to_id.items()
        }

        self.vocab_size = len(self.token_to_id)

    def encode(self, text):

        # Begin with individual characters
        tokens = list(text)

        # Apply learned merges in training order
        for pair in self.merges:
            tokens = self.merge_pair(tokens, pair)

        return tokens

    def encode_ids(self, text):

        tokens = self.encode(text)

        return [
            self.token_to_id.get(token, 0)
            for token in tokens
        ]

    def decode(self, tokens):

        return "".join(tokens)

    def decode_ids(self, token_ids):

        tokens = [
            self.id_to_token.get(token_id, "<UNK>")
            for token_id in token_ids
        ]

        return self.decode(tokens)
