"""Feature extraction: build the BPE merge table from the initial word vocabulary.

Reads the ingested word vocabulary (space-separated character sequences with
counts) and greedily merges the most frequent
adjacent symbol pair (Sennrich et al. 2016), writing the learned merges
to the merge table in order.

Usage:
    pdm run extract       # word vocabulary -> BPE merge table
"""

import json
import logging
from collections import Counter
from pathlib import Path

from llm.log import setup_logging

log = logging.getLogger(__name__)

VOCAB_PATH = Path("data/feature/vocab.json")
MERGES_PATH = Path("data/feature/merges.txt")
NUM_MERGES = 1000


def count_pairs(vocab: dict[str, int]) -> Counter[tuple[str, str]]:
    pairs: Counter[tuple[str, str]] = Counter()
    for word, count in vocab.items():
        symbols = word.split()
        for pair in zip(symbols, symbols[1:]):
            pairs[pair] += count
    return pairs


def merge_pair(pair: tuple[str, str], vocab: dict[str, int]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for word, count in vocab.items():
        symbols = word.split()
        out: list[str] = []
        i = 0
        while i < len(symbols):
            if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
                out.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        merged[" ".join(out)] = count
    return merged


def main() -> None:
    setup_logging()
    vocab: dict[str, int] = json.loads(VOCAB_PATH.read_text())

    merges: list[tuple[str, str]] = []
    for step in range(NUM_MERGES):
        pairs = count_pairs(vocab)
        if not pairs:
            break
        pair, freq = pairs.most_common(1)[0]
        merges.append(pair)
        vocab = merge_pair(pair, vocab)
        if step < 10 or (step + 1) % 100 == 0:
            log.info(f"merge {step + 1:4d}: {pair[0]} + {pair[1]}  (freq {freq})")

    MERGES_PATH.parent.mkdir(parents=True, exist_ok=True)
    MERGES_PATH.write_text("".join(f"{a} {b}\n" for a, b in merges))

    symbols = {s for word in vocab for s in word.split()}
    log.info(f"merges learned: {len(merges)}")
    log.info(f"symbol vocab size: {len(symbols)}")
    log.info(f"merges written to {MERGES_PATH}")
    log.debug("top 20 words after merging:")
    top = sorted(vocab.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
    for word, count in top:
        log.debug(f"  {count:6d}  {word}")


if __name__ == "__main__":
    main()
