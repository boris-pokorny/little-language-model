"""Data ingestion: read every PDF in the corpus folder once, write the plain
text and the initial word vocabulary.

The PDFs are read in sorted filename order. The text goes to CORPUS_PATH as
one line per page, lowercased words only; later stages read that file and
never touch the PDFs again. Each word also becomes a space-separated
character sequence terminated by the end-of-word marker </w>
(Sennrich et al. 2016), mapped to its corpus count, in VOCAB_PATH.

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


def extract_pages(pdf_path: Path) -> list[str]:
    """One string per page, reduced to lowercase words."""
    reader = PdfReader(pdf_path)
    return [" ".join(WORD_RE.findall((page.extract_text() or "").lower())) for page in reader.pages]


def build_vocab(words: list[str]) -> Counter[str]:
    return Counter(" ".join(word) + " </w>" for word in words)


def main() -> None:
    setup_logging()
    pages: list[str] = []
    for pdf in corpus_pdfs():
        pdf_pages = extract_pages(pdf)
        log.info(f"{pdf.name}: {len(pdf_pages)} pages")
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
