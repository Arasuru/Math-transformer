class CharTokenizer:
    def __init__(self):
        self.chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '=', '<PAD>', '<EOS>']

        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.chars)}

        self.pad_token_id = self.char_to_idx['<PAD>']
        self.eos_token_id = self.char_to_idx['<EOS>']

        self.vocab_size = len(self.chars)

    def encode(self, text):
        "encoding a string equation into a list of token ids"
        return [self.char_to_idx[char] for char in text if char in self.char_to_idx]

    def decode(self, token_ids):
        "decoding a list of token ids into a string equation"
        return ''.join([self.idx_to_char[idx] for idx in token_ids if idx not in [self.pad_token_id, self.eos_token_id]])


if __name__ == "__main__":
    tokenizer = CharTokenizer()
    test_equation = "12+34=46"
    encoded = tokenizer.encode(test_equation)
    decoded = tokenizer.decode(encoded)

    print(f"Original: {test_equation}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")