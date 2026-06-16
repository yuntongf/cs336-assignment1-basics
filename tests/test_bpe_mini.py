from cs336_basics.bpe import tokenize_pretok
from tests.adapters import run_train_bpe

from .common import FIXTURES_PATH


def test_bpe_mini():
    input_path = FIXTURES_PATH / "address.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )


def test_tokenize_pretok_recursive():
    merges: list[tuple[bytes, bytes]] = [(b"a", b"b"), (b"ab", b"c")]
    pretok = b"abcdd"
    toks = tokenize_pretok(pretok, merges)
    assert toks == [b"abc", b"d", b"d"]


def test_tokenize_pretok_order_matters():
    merges: list[tuple[bytes, bytes]] = [(b"b", b"c"), (b"a", b"b")]
    pretok = b"abc"
    toks = tokenize_pretok(pretok, merges)
    assert toks == [b"a", b"bc"]
