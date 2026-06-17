#bash

rm train.bin memray-flamegraph-train.html
uv run --with memray python -m memray run -o train.bin cs336_basics/train_bpe.py
uv run --with memray python -m memray flamegraph train.bin
open memray-flamegraph-train.html
