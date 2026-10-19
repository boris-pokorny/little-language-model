"""Model training: train a tiny transformer language model on the tokenized corpus.

Reads the tokenized corpus, trains TinyGPT (model.py) on next-token
prediction, saves the checkpoint and prints a few sampled continuations.

Hardware is chosen once in main() via backend.select_backend() and
passed down; the training loop itself is device-agnostic.

Usage:
    pdm run train [--steps N] [--device cpu|cuda|mps] [--prompt "..."]
"""

import argparse
import copy
import logging
import time
from dataclasses import asdict
from pathlib import Path

import torch

from llm.backend import Backend, select_backend
from llm.log import setup_logging
from llm.model import GPTConfig, TinyGPT
from llm.tokenize_text import TOKENS_PATH, tokenize

log = logging.getLogger(__name__)

MODEL_PATH = Path("data/model/model.pt")


def load_corpus() -> tuple[torch.Tensor, list[str]]:
    if not TOKENS_PATH.exists():
        raise SystemExit(f"{TOKENS_PATH} not found - run `pdm run tokenize` first")
    tokens = TOKENS_PATH.read_text().split()
    itos = sorted(set(tokens))
    stoi = {tok: i for i, tok in enumerate(itos)}
    data = torch.tensor([stoi[tok] for tok in tokens], dtype=torch.long)
    return data, itos


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(
    model: TinyGPT, data: torch.Tensor, backend: Backend, batch_size: int, iters: int = 25
) -> float:
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, model.cfg.block_size, batch_size)
        with backend.autocast():
            _, loss = model(backend.to(x), backend.to(y))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train(
    model: TinyGPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    backend: Backend,
    *,
    steps: int,
    batch_size: int = 32,
    lr: float = 3e-4,
    eval_interval: int = 100,
    patience: int = 5,
) -> float:
    """Train until `steps` or until val loss hasn't improved for `patience`
    evals; the model is left holding the best-val weights. Returns best val."""
    model = backend.to(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    start = time.perf_counter()
    best_val, best_state, since_best = float("inf"), copy.deepcopy(model.state_dict()), 0
    for step in range(steps + 1):
        if step % eval_interval == 0 or step == steps:
            tr = estimate_loss(model, train_data, backend, batch_size)
            va = estimate_loss(model, val_data, backend, batch_size)
            if va < best_val:
                best_val, best_state, since_best = va, copy.deepcopy(model.state_dict()), 0
                mark = "*"
            else:
                since_best += 1
                mark = " "
            log.info(f"step {step:5d}  train {tr:.3f}  val {va:.3f} {mark}")
            if since_best >= patience:
                log.info(f"early stop: no val improvement in {patience} evals")
                break
        if step == steps:
            break
        x, y = get_batch(train_data, model.cfg.block_size, batch_size)
        with backend.autocast():
            _, loss = model(backend.to(x), backend.to(y))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    backend.synchronize()
    model.load_state_dict(best_state)
    log.info(f"best val {best_val:.3f}  ({time.perf_counter() - start:.1f}s)")
    return best_val


def detokenize(tokens: list[str]) -> str:
    return "".join(tokens).replace("</w>", " ").strip()


def sample(
    model: TinyGPT, itos: list[str], backend: Backend, prompt: str, n: int, length: int
) -> None:
    stoi = {tok: i for i, tok in enumerate(itos)}
    ids = [stoi[tok] for tok in tokenize(prompt) if tok in stoi] or [0]
    model.eval()
    log.debug(f'samples for prompt "{prompt}":')
    for _ in range(n):
        idx = backend.to(torch.tensor([ids], dtype=torch.long))
        out = model.generate(idx, length)[0].tolist()
        log.debug(f"  {detokenize([itos[i] for i in out])}")


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=5000, help="max steps; early stop may end sooner")
    ap.add_argument("--patience", type=int, default=5, help="evals without val improvement before stopping")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--device", default=None, help="cpu, cuda[:N] or mps (default: auto)")
    ap.add_argument("--prompt", default="the secret of")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    backend = select_backend(args.device)

    data, itos = load_corpus()
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    log.info(f"tokens: {len(data)}  vocab: {len(itos)}")

    config = GPTConfig(vocab_size=len(itos), dropout=args.dropout)
    model = TinyGPT(config)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"params: {n_params / 1e6:.2f}M  initial loss ~ ln(vocab) = {torch.log(torch.tensor(len(itos))):.3f}")

    train(model, train_data, val_data, backend, steps=args.steps, patience=args.patience)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": asdict(config), "itos": itos}, MODEL_PATH)
    log.info(f"model written to {MODEL_PATH}")

    sample(model, itos, backend, args.prompt, n=3, length=40)


if __name__ == "__main__":
    main()
