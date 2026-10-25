"""Model evaluation: score the trained model on the frozen benchmarks, deterministically.

For every text in data/bench/*.txt the model's total negative log-likelihood is
computed with one sequential pass over non-overlapping block_size windows (no
random batches), and reported three ways:

    token_ppl   exp(NLL / tokens)      the usual number, but it depends on the
                                       tokenizer and so is NOT comparable once
                                       more data changes the BPE merges
    word_ppl    exp(NLL / words)       tokenizer-independent
    bpc         NLL / ln2 / characters tokenizer-independent

plus the OOV rate (benchmark tokens the model's vocabulary lacks; they are
dropped from scoring) and the in-distribution val loss on the training split.
The prompts in data/bench/prompts.txt are continued greedily and by seeded
sampling into data/metrics/samples.md for a qualitative diff.

Outputs: data/metrics/eval.json (numbers only, tracked by DVC as metrics),
data/metrics/eval_meta.json (provenance + the model's own meta), data/metrics/samples.md.

Usage:
    pdm run evaluate [--device cpu|cuda|mps]
"""

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from llm.backend import Backend, select_backend
from llm.chat import add_decoding_args, build_logits_filter, trim_partial_word
from llm.checkpoint import MODEL_PATH, load_checkpoint
from llm.log import setup_logging
from llm.model import LanguageModel
from llm.params import load_params
from llm.provenance import collect
from llm.tokenize_text import MERGES_PATH, WORD_RE, tokenize
from llm.train_model import TOKENS_PATH, detokenize, load_corpus, split_data

log = logging.getLogger(__name__)

BENCH_DIR = Path("data/bench")
METRICS_PATH = Path("data/metrics/eval.json")
META_PATH = Path("data/metrics/eval_meta.json")
SAMPLES_PATH = Path("data/metrics/samples.md")

DEFAULTS = {"batch_size": 32, "seed": 1337, "sample_length": 40, "prompts_path": "data/bench/prompts.txt"}


@torch.no_grad()
def total_nll(model: LanguageModel, ids: torch.Tensor, backend: Backend, batch_size: int) -> tuple[float, int]:
    """Sum of next-token NLL over one sequential pass; returns (nll, predicted tokens)."""
    model.eval()
    block = model.cfg.block_size
    windows = [ids[i : i + block + 1] for i in range(0, len(ids) - 1, block)]
    windows = [w for w in windows if len(w) > 1]
    nll, n = 0.0, 0
    for start in range(0, len(windows), batch_size):
        chunk = windows[start : start + batch_size]
        length = len(chunk[0])  # all full except possibly the last window
        full = [w for w in chunk if len(w) == length]
        rest = [w for w in chunk if len(w) != length]
        for group in (full, rest):
            if not group:
                continue
            xy = backend.to(torch.stack(group))
            with backend.autocast():
                logits, _ = model(xy[:, :-1])
            nll += F.cross_entropy(logits.flatten(0, 1).float(), xy[:, 1:].flatten(), reduction="sum").item()
            n += xy[:, 1:].numel()
    return nll, n


def score_text(model: LanguageModel, stoi: dict[str, int], text: str, backend: Backend, batch_size: int) -> dict[str, Any]:
    words = WORD_RE.findall(text.lower())
    tokens = tokenize(text)
    known = [stoi[t] for t in tokens if t in stoi]
    nll, n_pred = total_nll(model, torch.tensor(known, dtype=torch.long), backend, batch_size)
    n_chars = sum(len(w) + 1 for w in words)  # one separator per word
    return {
        "nll": nll,
        "tokens": len(tokens),
        "words": len(words),
        "chars": n_chars,
        "oov_rate": 1 - len(known) / max(len(tokens), 1),
        "token_ppl": math.exp(nll / max(n_pred, 1)),
        "word_ppl": math.exp(nll / max(len(words), 1)),
        "bpc": nll / math.log(2) / max(n_chars, 1),
    }


def val_loss(model: LanguageModel, backend: Backend, batch_size: int) -> float:
    """Deterministic mean NLL on the training-time validation split."""
    split = load_params("train", {"split": 0.9})["split"]
    data, _ = load_corpus()
    _, val = split_data(data, split)
    nll, n = total_nll(model, val, backend, batch_size)
    return nll / max(n, 1)


def bench_files() -> list[Path]:
    return sorted(p for p in BENCH_DIR.glob("*.txt") if p.name != "prompts.txt")


def evaluate_model(model: LanguageModel, itos: list[str], backend: Backend, batch_size: int) -> dict[str, Any]:
    stoi = {tok: i for i, tok in enumerate(itos)}
    loss = val_loss(model, backend, batch_size)
    metrics: dict[str, Any] = {"val_loss": loss, "val_ppl": math.exp(loss), "bench": {}}
    for path in bench_files():
        metrics["bench"][path.stem] = score_text(model, stoi, path.read_text(), backend, batch_size)
    return metrics


@torch.no_grad()
def greedy(model: LanguageModel, idx: torch.Tensor, length: int, logits_filter) -> list[int]:
    for _ in range(length):
        logits, _ = model(idx[:, -model.cfg.block_size :])
        logits = logits_filter(idx, logits[:, -1, :])
        idx = torch.cat([idx, logits.argmax(dim=-1, keepdim=True)], dim=1)
    return idx[0].tolist()


def write_samples(model: LanguageModel, itos: list[str], backend: Backend, prompts: list[str], length: int, seed: int) -> None:
    stoi = {tok: i for i, tok in enumerate(itos)}
    ap = argparse.ArgumentParser()
    add_decoding_args(ap)
    logits_filter = build_logits_filter(itos, ap.parse_args([]))
    lines = ["# Benchmark prompt continuations", "", f"Decoding: default chat filters, seed {seed}, {length} tokens.", ""]
    for prompt in prompts:
        ids = [stoi[tok] for tok in tokenize(prompt) if tok in stoi]
        if not ids:
            lines += [f"## {prompt}", "", "(no known tokens)", ""]
            continue
        idx = backend.to(torch.tensor([ids], dtype=torch.long))
        g = detokenize(trim_partial_word([itos[i] for i in greedy(model, idx, length, logits_filter)]))
        torch.manual_seed(seed)
        s = model.generate(idx, length, temperature=0.8, logits_filter=logits_filter)[0].tolist()
        s = detokenize(trim_partial_word([itos[i] for i in s]))
        lines += [f"## {prompt}", "", f"- greedy: {g}", f"- sampled: {s}", ""]
    SAMPLES_PATH.write_text("\n".join(lines))


def main() -> None:
    setup_logging()
    params = load_params("evaluate", DEFAULTS)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default=None, help="cpu, cuda[:N] or mps (default: auto)")
    args = ap.parse_args()

    torch.manual_seed(params["seed"])
    backend = select_backend(args.device)
    model, ckpt = load_checkpoint(MODEL_PATH, backend)
    itos, ckpt_meta = ckpt["itos"], ckpt.get("meta", {})

    metrics = evaluate_model(model, itos, backend, params["batch_size"])
    log.info(f"val loss {metrics['val_loss']:.3f}  perplexity {metrics['val_ppl']:.1f}")
    for name, m in metrics["bench"].items():
        log.info(f"{name}: word_ppl {m['word_ppl']:.1f}  bpc {m['bpc']:.3f}  token_ppl {m['token_ppl']:.1f}  oov {m['oov_rate']:.2%}")

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")
    inputs = {"model": MODEL_PATH, "merges": MERGES_PATH, "tokens": TOKENS_PATH, **{p.name: p for p in bench_files()}}
    META_PATH.write_text(json.dumps({"provenance": collect(inputs), "model": ckpt_meta}, indent=2) + "\n")

    prompts_path = Path(params["prompts_path"])
    prompts = [l.strip() for l in prompts_path.read_text().splitlines() if l.strip()] if prompts_path.exists() else []
    write_samples(model, itos, backend, prompts, params["sample_length"], params["seed"])
    log.info(f"metrics written to {METRICS_PATH}, {META_PATH}; samples to {SAMPLES_PATH}")


if __name__ == "__main__":
    main()
