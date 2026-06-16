import pickle
from collections.abc import Iterable

import regex as re
from sympy.solvers.ode.single import Iterator

from cs336_basics.bpe import PAT, apply_merge


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.inverted_vocab: dict[bytes, int] = {b: i for i, b in self.vocab.items()}
        for special_tok in special_tokens or []:
            if special_tok.encode() in self.inverted_vocab:
                continue
            idx = len(self.vocab)
            self.vocab[idx] = special_tok.encode()
            self.inverted_vocab[special_tok.encode()] = idx

        self.merges: dict[tuple[bytes, bytes], int] = {bp: i for i, bp in enumerate(merges)}
        self.special_tokens = set(special_tokens or [])

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        res = [tok for tok in self.encode_iterable([text])]
        return res

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for s in iterable:
            pretoks = self._pretokenize(s)
            for pretok in pretoks:
                if pretok in self.special_tokens:
                    yield self.inverted_vocab[pretok.encode()]
                    continue

                toks = self._apply_merge_recursive([bytes([b]) for b in pretok.encode()])
                for tok in toks:
                    yield self.inverted_vocab[tok]

    def _pretokenize(self, s: str) -> list[str]:
        toks = []
        sorted_special_tokens = sorted([tok for tok in self.special_tokens], key=lambda a: len(a), reverse=True)
        sections = (
            re.split("(" + "|".join([re.escape(tok) for tok in sorted_special_tokens]) + ")", s)
            if self.special_tokens
            else [s]
        )
        for i, section in enumerate(sections):
            if i % 2 == 1:  # captured special tokens
                toks.append(section)
            else:
                for match in re.finditer(PAT, section):
                    toks.append(match.group(0))
        return toks

    def _apply_merge_recursive(self, toks: list[bytes]) -> list[bytes]:
        merged_toks = toks
        while True:
            valid_merges = []
            i = 0
            while i < len(merged_toks) - 1:
                bp = merged_toks[i], merged_toks[i + 1]
                # print(bp)
                if bp in self.merges:
                    order = self.merges[bp]
                    valid_merges.append((order, bp))
                i += 1

            if len(valid_merges) == 0:
                break
            m = min(valid_merges)[1]
            merged_toks = apply_merge(merged_toks, m)

        return merged_toks

    def decode(self, ids: list[int]) -> str:
        toks: list[bytes] = [self.vocab[id] for id in ids]
        raw_str = b""
        for tok in toks:
            raw_str += tok
        return raw_str.decode(errors="replace")
