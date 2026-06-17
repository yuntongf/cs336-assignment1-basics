import multiprocessing as mp
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import wraps
from typing import Any, BinaryIO, Callable

import regex as re

from cs336_basics.heapmap import HeapMap, HeapMapMember

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
CHUNK_SIZE = 1024


def timer(func: Callable) -> Callable:
    """Decorator that prints the execution time of a function."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        print(f"{func.__name__}() took {elapsed:.6f} seconds")
        return result

    return wrapper


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = 2**28
    num_chunks = (file_size + chunk_size - 1) // chunk_size

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize_chunk(chunk: str, special_tokens: list[str]) -> dict[bytes, int]:
    tok_to_freq = defaultdict(int)
    escaped_special_tokens = [re.escape(tok) for tok in special_tokens]
    for section in re.split("|".join(escaped_special_tokens), chunk):
        for match in re.finditer(PAT, section):
            tok_to_freq[match.group(0).encode()] += 1
    return tok_to_freq


def merge_pretokens(pretoken_maps: list[dict[bytes, int]]) -> dict[bytes, int]:
    res = defaultdict(int)
    for m in pretoken_maps:
        for pretok_bytes, count in m.items():
            res[pretok_bytes] += count
    return res


def count_bp_freq(pretok_map: dict[bytes, int]) -> dict[tuple[bytes, bytes], tuple[int, set[bytes]]]:
    bp_freq = {}
    for tok, count in pretok_map.items():
        for i in range(len(tok) - 1):
            bp = (tok[i : i + 1], tok[i + 1 : i + 2])
            if bp in bp_freq:
                (old_count, pretoks) = bp_freq[bp]
                pretoks.add(tok)
                bp_freq[bp] = (count + old_count, pretoks)
            else:
                bp_freq[bp] = (count, set([tok]))

    return bp_freq


def apply_merge(toks: list[bytes], merge: tuple[bytes, bytes]) -> list[bytes]:
    new_toks = []
    i = 0
    while i < len(toks):
        if i == len(toks) - 1 or merge != (toks[i], toks[i + 1]):
            new_toks.append(toks[i])
        else:
            new_toks.append(toks[i] + toks[i + 1])
            i += 1
        i += 1
    return new_toks


type BP = tuple[bytes, bytes]


@dataclass
class BPEntry(HeapMapMember[BP, tuple[int, set[bytes]]]):
    bp: BP
    v: tuple[int, set[bytes]]

    def __lt__(self, other) -> bool:
        if self.v[0] == other.v[0]:
            return self.bp > other.bp
        return self.v[0] > other.v[0]

    def key(self) -> BP:
        return self.bp

    def val(self) -> tuple[int, set[bytes]]:
        return self.v


def merge_tokens(
    pretok_map: dict[bytes, tuple[list[bytes], int]],
    bp_map: dict[tuple[bytes, bytes], tuple[int, set[bytes]]],
    special_tokens: list[str],
    vocab_size: int,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    merges: list[tuple[bytes, bytes]] = []
    vocab: set[bytes] = set([bytes([i]) for i in range(256)] + [tok.encode() for tok in special_tokens])
    bp_heap_map: HeapMap[BP, tuple[int, set[bytes]]] = HeapMap(list(bp_map.items()), BPEntry)

    while len(vocab) < vocab_size:
        bp_to_merge, (_, affected_pretoks) = bp_heap_map.pop()

        merges.append(bp_to_merge)
        vocab.add(bp_to_merge[0] + bp_to_merge[1])

        for pretok_bytes in affected_pretoks:
            (toks, pretok_count) = pretok_map[pretok_bytes]
            # remove original counts
            for i in range(len(toks) - 1):
                bp = (toks[i], toks[i + 1])
                if bp in bp_heap_map:
                    (count, old_pretoks) = bp_heap_map[bp]
                    old_pretoks.discard(pretok_bytes)
                    bp_heap_map.update(bp, (count - pretok_count, old_pretoks))

            # add new counts
            new_toks = apply_merge(toks, bp_to_merge)
            for i in range(len(new_toks) - 1):
                bp = (new_toks[i], new_toks[i + 1])
                if bp not in bp_heap_map:
                    bp_heap_map.insert(bp, (pretok_count, set([pretok_bytes])))
                else:
                    (old_count, old_pretoks) = bp_heap_map[bp]
                    old_pretoks.add(pretok_bytes)
                    bp_heap_map.update(bp, (old_count + pretok_count, old_pretoks))

            pretok_map[pretok_bytes] = (new_toks, pretok_count)

    vocab_idx = dict(zip(range(len(vocab)), vocab))

    return vocab_idx, merges


def pretokenize(start, end, input_path, special_tokens):
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        return pretokenize_chunk(chunk, special_tokens)


def bpe_encode(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    start_time = time.time()
    with open(input_path, "rb") as f:
        num_processes = mp.cpu_count()
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    pool = mp.Pool(num_processes)

    futs = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        fut = pool.apply_async(
            pretokenize,
            args=(start, end, input_path, special_tokens),
        )
        futs.append(fut)

    pool.close()
    pool.join()

    merged_pretok_map = {}
    for fut in futs:
        merged_pretok_map = merge_pretokens([fut.get(), merged_pretok_map])

    read_and_pretok = time.time()
    print(f"read and pretok: {read_and_pretok - start_time}")

    merge_pretok = time.time()
    print(f"merge pretok map {merge_pretok - read_and_pretok}")

    bp_map = count_bp_freq(merged_pretok_map)
    build_bp_map = time.time()
    print(f"build bp map {build_bp_map - merge_pretok}")

    pretok_map = {
        pretok_bytes: ([bytes([b]) for b in pretok_bytes], count) for pretok_bytes, count in merged_pretok_map.items()
    }

    vocab, merged = merge_tokens(pretok_map, bp_map, special_tokens, vocab_size)
    merge_toks = time.time()
    print(f"merge toks {merge_toks - build_bp_map}")

    return vocab, merged
