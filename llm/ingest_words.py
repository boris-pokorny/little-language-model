"""Data ingestion: read every PDF in the corpus folder once, write the plain
text and the initial word vocabulary.

The PDFs are read in sorted filename order. The text goes to CORPUS_PATH as
one line per page, lowercased words only; later stages read that file and
never touch the PDFs again. Each word also becomes a space-separated
character sequence terminated by the end-of-word marker </w>
(Sennrich et al. 2016), mapped to its corpus count, in VOCAB_PATH.

Pages listed under `ingest.exclude_pages` in params.yaml (1-based inclusive
ranges per file name) are skipped: the held-out benchmark text in data/bench/
was carved from them and must never be trained on. data/bench/ itself is not
read here.

Usage:
    pdm run ingest        # data/corpus/*.pdf -> corpus text + word vocabulary
"""

import json
import logging
import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

from llm.log import setup_logging
from llm.params import load_params

log = logging.getLogger(__name__)

CORPUS_DIR = Path("data/corpus")
CORPUS_PATH = Path("data/feature/corpus.txt")
VOCAB_PATH = Path("data/feature/vocab.json")

WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def corpus_pdfs() -> list[Path]:
    pdfs = sorted(CORPUS_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"no PDF files in {CORPUS_DIR}/ - put at least one there first")
    return pdfs


def excluded_pages(ranges: list[list[int]]) -> set[int]:
    """Expand [[first, last], ...] (1-based, inclusive) into a set of page numbers."""
    return {n for first, last in ranges for n in range(first, last + 1)}


def extract_pages(pdf_path: Path, skip: set[int] = frozenset()) -> list[str]:
    """One string per page, reduced to lowercase words; 1-based pages in `skip` are dropped."""
    reader = PdfReader(pdf_path)
    return [
        " ".join(WORD_RE.findall((page.extract_text() or "").lower()))
        for n, page in enumerate(reader.pages, 1)
        if n not in skip
    ]


def build_vocab(words: list[str]) -> Counter[str]:
    return Counter(" ".join(word) + " </w>" for word in words)


def main() -> None:
    setup_logging()
    exclude: dict[str, list[list[int]]] = load_params("ingest", {"exclude_pages": {}})["exclude_pages"] or {}
    pdfs = corpus_pdfs()
    for stale in sorted(set(exclude) - {pdf.name for pdf in pdfs}):
        log.warning(f"ingest.exclude_pages in params.yaml names {stale!r}, which is not in {CORPUS_DIR}/ - ignored")
    pages: list[str] = []
    for pdf in pdfs:
        skip = excluded_pages(exclude.get(pdf.name, []))
        pdf_pages = extract_pages(pdf, skip)
        log.info(f"{pdf.name}: {len(pdf_pages)} pages" + (f" ({len(skip)} excluded)" if skip else ""))
        pages.extend(pdf_pages)

    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CORPUS_PATH.write_text("\n".join(pages) + "\n")
    log.info(f"corpus written to {CORPUS_PATH}")

    vocab = build_vocab(" ".join(pages).split())
    ordered = dict(sorted(vocab.items(), key=lambda kv: (-kv[1], kv[0])))
    VOCAB_PATH.write_text(json.dumps(ordered, indent=2) + "\n")

    total = sum(vocab.values())
    log.info(f"total words:  {total}")
    log.info(f"unique words: {len(vocab)}")
    log.info(f"vocab written to {VOCAB_PATH}")
    log.debug("top 20:")
    for entry, count in list(ordered.items())[:20]:
        log.debug(f"  {count:6d}  {entry}")


if __name__ == "__main__":
    main()
