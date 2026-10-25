"""Checkpoint file format and the save/load pair that implements it.

model.pt is a torch.save dict with keys:
    arch: Registry name of the architecture.
    config: `asdict` of the model's config dataclass.
    model: The state dict.
    itos: Token index -> string table of the tokenizer.
    meta: Provenance, params and metrics; also written as model.json.
"""

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from llm.backend import Backend
from llm.model import LanguageModel, architecture

MODEL_PATH = Path("data/model/model.pt")
MODEL_CARD_PATH = Path("data/model/model.json")

log = logging.getLogger(__name__)


def save_checkpoint(
    path: Path, model: LanguageModel, arch: str, itos: list[str], meta: dict[str, Any]
) -> None:
    """Writes a checkpoint, creating parent directories as needed.

    Args:
        path: Destination file.
        model: Model whose config and weights are saved.
        arch: Registry name of the model's architecture.
        itos: Token index -> string table.
        meta: Arbitrary JSON-serializable record stored alongside the weights.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"arch": arch, "config": asdict(model.cfg), "model": model.state_dict(), "itos": itos, "meta": meta},
        path,
    )


def load_checkpoint(path: Path, backend: Backend) -> tuple[LanguageModel, dict[str, Any]]:
    """Rebuilds a model from a checkpoint and logs what was loaded.

    Args:
        path: Checkpoint file written by `save_checkpoint`.
        backend: Supplies the device the model and tensors are placed on.

    Returns:
        Tuple (model, ckpt): the model in eval mode on the backend's device, and
        the raw checkpoint dict.

    Raises:
        SystemExit: If `path` does not exist.
    """
    if not path.exists():
        raise SystemExit(f"{path} not found - run `pdm run train` first")
    ckpt = torch.load(path, map_location=backend.device)
    arch = ckpt["arch"]
    cls = architecture(arch)
    model = backend.to(cls(cls.config_cls(**ckpt["config"])))
    model.load_state_dict(ckpt["model"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"model: {arch} ({cls.__name__})  params: {n_params / 1e6:.2f}M  from {path}")
    return model, ckpt
