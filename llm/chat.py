"""Model evaluation and inference: evaluate the trained model and sample from it interactively.

Loads the trained checkpoint (from the training stage), reports the same
deterministic val loss / benchmark numbers as `pdm run evaluate`, then reads
prompts from stdin and prints continuations. The model is a plain next-token LM, so it continues your
prompt rather than answering it.

Decoding filters (all on by default; see --help to relax them):
    lexicon              only allow tokens that keep the current word a prefix of
                         a real word from the ingested vocabulary (guarantees real words);
                         --min-count C ignores words seen fewer than C times
                         (drops extraction junk like the "O" in "O. Henry")
    --repeat-penalty P   divide logits of tokens seen in the last 64 by P (>1)
    --no-repeat N        never re-emit a token N-gram already in the sequence
    --top-k K            sample only from the K most likely remaining tokens
Filters are applied in that order; one that would leave no token is skipped.

Usage:
    pdm run chat [--device cpu|cuda|mps] [--length N] [--temperature T] [--no-eval]  (--help for all decoding knobs)
"""

import argparse
import json
import logging
import math
from typing import Callable

import torch

from llm.backend import Backend, select_backend
from llm.checkpoint import MODEL_PATH, load_checkpoint
from llm.ingest_words import VOCAB_PATH
from llm.log import setup_logging
from llm.model import LanguageModel
from llm.tokenize_text import tokenize, tokenize_word
from llm.train_model import detokenize

log = logging.getLogger(__name__)

LogitsFilter = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def top_k_filter(k: int) -> LogitsFilter:
    def f(idx: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        kth = torch.topk(logits, k, dim=-1).values[:, -1:]
        return logits.masked_fill(logits < kth, float("-inf"))

    return f


def repeat_penalty_filter(penalty: float, window: int = 64) -> LogitsFilter:
    def f(idx: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        recent = idx[:, -window:]
        seen = torch.gather(logits, 1, recent)
        seen = torch.where(seen > 0, seen / penalty, seen * penalty)
        return logits.scatter(1, recent, seen)

    return f


def no_repeat_ngram_filter(n: int) -> LogitsFilter:
    def f(idx: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.clone()
        for row, seq in enumerate(idx.tolist()):
            if len(seq) < n:
                continue
            prefix = tuple(seq[-(n - 1) :]) if n > 1 else ()
            banned = {
                seq[i + n - 1] for i in range(len(seq) - n + 1) if tuple(seq[i : i + n - 1]) == prefix
            }
            if banned:
                logits[row, list(banned)] = float("-inf")
        return logits

    return f


def lexicon_filter(itos: list[str], min_count: int) -> LogitsFilter:
    """Allow only tokens that extend the current partial word toward a known word.

    Builds, for every prefix of every word's tokenization, the set of token ids
    that may follow it. A word boundary is any token ending in </w>.
    """
    stoi = {tok: i for i, tok in enumerate(itos)}
    words = [
        w.replace(" ", "").removesuffix("</w>")
        for w, count in json.loads(VOCAB_PATH.read_text()).items()
        if count >= min_count
    ]
    allowed: dict[tuple[int, ...], set[int]] = {}
    for word in words:
        ids = tuple(stoi[t] for t in tokenize_word(word) if t in stoi)
        for i in range(len(ids)):
            allowed.setdefault(ids[:i], set()).add(ids[i])
    tables = {prefix: torch.tensor(sorted(ids)) for prefix, ids in allowed.items()}
    ends_word = [tok.endswith("</w>") for tok in itos]

    def f(idx: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        out = torch.full_like(logits, float("-inf"))
        for row, seq in enumerate(idx.tolist()):
            partial: list[int] = []
            for t in reversed(seq):
                if ends_word[t]:
                    break
                partial.append(t)
            ok = tables.get(tuple(reversed(partial)))
            if ok is None:  # partial word not in lexicon: force a boundary
                ok = torch.tensor([i for i, e in enumerate(ends_word) if e])
            ok = ok.to(logits.device)
            out[row, ok] = logits[row, ok]
        return out

    return f


def trim_partial_word(tokens: list[str]) -> list[str]:
    """Drop a trailing word cut off by the length limit."""
    while tokens and not tokens[-1].endswith("</w>"):
        tokens.pop()
    return tokens


def add_decoding_args(ap: argparse.ArgumentParser) -> None:
    """Device + decoding-filter knobs shared by chat.py and agent.py."""
    ap.add_argument("--device", default=None, help="cpu, cuda[:N] or mps (default: auto)")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--no-lexicon", action="store_true", help="allow any token sequence")
    ap.add_argument("--min-count", type=int, default=2, help="lexicon: min corpus count of a word")
    ap.add_argument("--repeat-penalty", type=float, default=1.3, help="1.0 = off")
    ap.add_argument("--no-repeat", type=int, default=3, help="block repeated N-grams, 0 = off")
    ap.add_argument("--top-k", type=int, default=40, help="0 = off")


def load_model(backend: Backend) -> tuple[LanguageModel, list[str]]:
    model, ckpt = load_checkpoint(MODEL_PATH, backend)
    return model, ckpt["itos"]


def build_logits_filter(itos: list[str], args: argparse.Namespace) -> LogitsFilter:
    """Chain the filters selected by add_decoding_args; one that would leave
    no token is skipped."""
    filters: list[LogitsFilter] = []
    if not args.no_lexicon:
        filters.append(lexicon_filter(itos, args.min_count))
    if args.repeat_penalty != 1.0:
        filters.append(repeat_penalty_filter(args.repeat_penalty))
    if args.no_repeat:
        filters.append(no_repeat_ngram_filter(args.no_repeat))
    if args.top_k:
        filters.append(top_k_filter(args.top_k))

    def logits_filter(idx: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        for f in filters:
            candidate = f(idx, logits)
            if torch.isfinite(candidate).any(dim=-1).all():
                logits = candidate
        return logits

    return logits_filter


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_decoding_args(ap)
    ap.add_argument("--length", type=int, default=60, help="tokens to generate per prompt")
    ap.add_argument("--no-eval", action="store_true", help="skip validation pass")
    args = ap.parse_args()

    backend = select_backend(args.device)
    model, itos = load_model(backend)
    stoi = {tok: i for i, tok in enumerate(itos)}

    if not args.no_eval:
        from llm.evaluate import evaluate_model  # lazy: evaluate imports this module

        metrics = evaluate_model(model, itos, backend, batch_size=32)
        log.info(f"val loss {metrics['val_loss']:.3f}  perplexity {metrics['val_ppl']:.1f}  (uniform = {len(itos)})")
        for name, m in metrics["bench"].items():
            log.info(f"{name}: word_ppl {m['word_ppl']:.1f}  bpc {m['bpc']:.3f}")

    logits_filter = build_logits_filter(itos, args)

    print("\ntype a prompt, empty line or Ctrl-D to quit")
    while True:
        try:
            prompt = input("> ").strip()
        except EOFError:
            break
        if not prompt:
            break
        ids = [stoi[tok] for tok in tokenize(prompt) if tok in stoi]
        if not ids:
            print("  (no known tokens in prompt)")
            continue
        idx = backend.to(torch.tensor([ids], dtype=torch.long))
        out = model.generate(
            idx, args.length, temperature=args.temperature, logits_filter=logits_filter
        )[0].tolist()
        print(f"  {detokenize(trim_partial_word([itos[i] for i in out]))}\n")


if __name__ == "__main__":
    main()
