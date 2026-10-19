"""Feature transformation: tokenize text using the learned merge rules.

Reads the extracted BPE merge table and applies the merges
in learned order to each word: start from characters + </w>, then
repeatedly merge the adjacent pair with the lowest merge rank.

Usage:
    pdm run tokenize                  # corpus text (from ingest) -> token file
    pdm run tokenize some text        # -> stdout
    echo "some text" | pdm run tokenize -
"""

import logging
import re
import sys
from functools import lru_cache
from pathlib import Path

from llm.ingest_words import CORPUS_PATH
from llm.log import setup_logging

log = logging.getLogger(__name__)

MERGES_PATH = Path("data/feature/merges.txt")
TOKENS_PATH = Path("data/feature/tokens.txt")

WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")

RANKS: dict[tuple[str, str], int] = {
    tuple(line.split()): rank
    for rank, line in enumerate(MERGES_PATH.read_text().splitlines())
}


@lru_cache(maxsize=None)
def tokenize_word(word: str) -> tuple[str, ...]:
    symbols = list(word) + ["</w>"]
    while len(symbols) > 1:
        pairs = zip(symbols, symbols[1:])
        best = min(pairs, key=lambda p: RANKS.get(p, len(RANKS)))
        if best not in RANKS:
            break
        out: list[str] = []
        i = 0
        while i < len(symbols):
            if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == best:
                out.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        symbols = out
    return tuple(symbols)


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for word in WORD_RE.findall(text.lower()):
        tokens.extend(tokenize_word(word))
    return tokens


def tokenize_corpus() -> None:
    """Tokenize the corpus text page by page (one line per page) into TOKENS_PATH."""
    if not CORPUS_PATH.exists():
        raise SystemExit(f"{CORPUS_PATH} not found - run `pdm run ingest` first")
    pages = CORPUS_PATH.read_text().splitlines()
    total = 0
    TOKENS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TOKENS_PATH.open("w") as out:
        for i, page in enumerate(pages, 1):
            tokens = tokenize(page)
            out.write(" ".join(tokens) + "\n")
            total += len(tokens)
            if i % 25 == 0 or i == len(pages):
                log.info(f"  page {i:4d}/{len(pages)}  {total:7d} tokens")
    log.info(f"tokens written to {TOKENS_PATH}")


def main() -> None:
    setup_logging()
    if len(sys.argv) == 1:
        tokenize_corpus()
        return
    text = sys.stdin.read() if sys.argv[1:] == ["-"] else " ".join(sys.argv[1:])
    tokens = tokenize(text)
    print(" ".join(tokens))
    log.info(f"{len(tokens)} tokens")


if __name__ == "__main__":
    main()
