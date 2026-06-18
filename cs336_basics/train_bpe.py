import pathlib
import pickle

from bpe import bpe_encode

TINYSTORIES_FILE = "TinyStoriesV2-GPT4-train.txt"
OPEN_WEB_TEXT_FILE = "owt_train.txt"

VOCAB_FILE = "vocab.pkl"
MERGES_FILE = "merges.pkl"

if __name__ == "__main__":
    input_path = pathlib.Path("data") / TINYSTORIES_FILE
    output_path = pathlib.Path("out")

    vocab, merges = bpe_encode(
        input_path=input_path,
        vocab_size=32_000,
        special_tokens=["<|endoftext|>"],
    )

    with open(output_path / MERGES_FILE, "wb") as f:
        pickle.dump(merges, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(output_path / VOCAB_FILE, "wb") as f:
        pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
