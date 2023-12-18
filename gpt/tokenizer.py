import json
import os

class CharacterTokenizer:
    def __init__(self, data=None, pad_token=None, eos_token=None):
        self.filepath = "character_tokenizer.json"
        if not os.path.exists(self.filepath):
            assert data is not None, "Please provide data to learn tokenizer."
            assert pad_token is not None, "Please provide pad_token to learn tokenizer."
            assert eos_token is not None, "Please provide pad_token to learn tokenizer."
            self.learn_tokenizer(data, pad_token, eos_token)
            self.save_tokenizer()
            print(f"Tokenizer learned and saved to disk.")
        else:
            print(f"Loading pretrained tokenizer from disk.")
            self.load_tokenizer()

    def learn_tokenizer(self, data, pad_token, eos_token):
        vocab = list(set(data))
        vocab.append(eos_token)
        # create a mapping from character to index.
        self.char_to_idx = {ch:i+1 for i,ch in enumerate(vocab)}
        self.char_to_idx["[PAD]"] = pad_token
        # create a mapping from index to character.
        self.idx_to_char = {i+1:ch for i,ch in enumerate(vocab)}
        self.idx_to_char[pad_token] = "[PAD]"

    def tokenize(self, data, add_eos=True):
        tokens = [self.char_to_idx[ch] for ch in data]
        if add_eos:
            tokens.append(self.char_to_idx["<EOS>"])
        return tokens

    def detokenize(self, data):
        chars = [self.idx_to_char[idx] for idx in data]
        if "<EOS>" in chars:
            return "".join(chars[:chars.index("<EOS>") + 1])
        else:
            return "".join(chars)
    
    def save_tokenizer(self):
        with open(self.filepath, "w") as f:
            json.dump(self.char_to_idx, f)

    def vocab_size(self):
        return len(self.char_to_idx)

    def load_tokenizer(self):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError("Pretrained tokenizer not found. Please run learn_tokenizer() first.")
        with open(self.filepath, "r") as f:
            self.char_to_idx = json.load(f)
        self.idx_to_char = {idx:ch for ch,idx in self.char_to_idx.items()}
