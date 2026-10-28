"""Closed word-level tokenizer: one token per word / number / variable / symbol."""

from .vocab import LETTERS, MOD, PUNCT, SPECIALS, TEMPLATE_WORDS, all_words


class Tokenizer:
    def __init__(self):
        words = sorted(all_words() | set(TEMPLATE_WORDS))
        numbers = [str(i) for i in range(MOD)]
        clash = set(words) & (set(LETTERS) | set(numbers) | set(PUNCT) | set(SPECIALS))
        assert not clash, f"vocabulary clash: {clash}"
        self.itos = list(SPECIALS) + numbers + list(LETTERS) + list(PUNCT) + words
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        assert len(self.stoi) == len(self.itos)

        self.pad, self.bos, self.eos, self.sol, self.mask, self.null = (self.stoi[s] for s in SPECIALS)
        self.num_ids = [self.stoi[n] for n in numbers]
        self.letter_ids = [self.stoi[c] for c in LETTERS]
        self.num_of = {self.stoi[n]: int(n) for n in numbers}  # token id -> int value
        self.letter_set = set(self.letter_ids)
        self.special_ids = [self.stoi[s] for s in SPECIALS]
        # Tokens that can never be produced by a fix / fill.
        self.no_fix_ids = [self.pad, self.bos, self.sol, self.mask, self.null, self.eos]
        self.no_fill_ids = [self.pad, self.bos, self.sol, self.mask, self.eos]

    def __len__(self):
        return len(self.itos)

    def num(self, v):
        return self.num_ids[v]

    def encode(self, words):
        return [self.stoi[w] for w in words]

    def decode(self, ids, skip_pad=True):
        out = []
        for i in ids:
            if skip_pad and i == self.pad:
                continue
            out.append(self.itos[i])
        return " ".join(out)

    def __getitem__(self, w):
        return self.stoi[w]
