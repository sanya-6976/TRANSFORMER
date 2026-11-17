
class CharacterTokenizer:

    def __init__(self, text):

        self.text = text

        # Reserve IDs for special tokens
        self.special_tokens = [
            "<PAD>",
            "<UNK>",
            "<BOS>",
            "<EOS>"
        ]

        self.build_vocab()

    # Build vocabulary
    def build_vocab(self):

        # Create special token mappings
        self.special_to_id = {
            token: index
            for index, token in enumerate(self.special_tokens)
        }

        characters = sorted(set(self.text))

        self.char_to_id = {
            char: index
            for index, char in enumerate(characters)
        }

        self.id_to_token = {
            index: token
            for index, token in enumerate(self.special_tokens + list(characters))
        }
        self.id_to_token.update({
            index: char
            for char, index in self.char_to_id.items()
        })

        # Store special token IDs
        self.pad_id = self.special_to_id["<PAD>"]
        self.unk_id = self.special_to_id["<UNK>"]
        self.bos_id = self.special_to_id["<BOS>"]
        self.eos_id = self.special_to_id["<EOS>"]

        self.vocab_size = len(self.id_to_token)

    # Convert text into token IDs
    def encode(self, text , add_bos = False , add_eos = False):

        encoded = [
            self.char_to_id.get(char, self.unk_id)
            for char in text
        ]

        if add_bos:
            encoded.insert(0 , self.bos_id)

        if add_eos:
            encoded.append(self.eos_id)

        

        return encoded

    # Convert token IDs back into text
    def decode(self, token_ids , skip_special_tokens = True):

        decoded = []

        for token_id in token_ids:

            token = self.id_to_token[int(token_id)]

            if token in ["<PAD>", "<BOS>", "<EOS>"]:

                if not skip_special_tokens:
                    decoded.append(token)

            elif token == "<UNK>":

                decoded.append("�")

            else:
                decoded.append(token)

        return "".join(decoded)



    def truncate(self, token_ids, max_length):

        return token_ids[:max_length]


    def pad_sequences(self, sequences, max_length=None):

        # Find longest sequence if length is not supplied
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)

        padded_sequences = []

        for sequence in sequences:

            # Truncate sequences that are too long
            sequence = sequence[:max_length]

            # Calculate required padding
            padding_length = max_length - len(sequence)

            # Add PAD tokens
            padded_sequence = (
                sequence +
                [self.pad_id] * padding_length
            )

            padded_sequences.append(padded_sequence)

        return padded_sequences
