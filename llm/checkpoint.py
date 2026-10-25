"""Checkpoint file format, in one place.

model.pt holds {"arch", "config", "model", "itos", "meta"}; checkpoints written
before the `arch` key existed are TinyGPT and load as such.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from llm.backend import Backend
from llm.model import LanguageModel, architecture

MODEL_PATH = Path("data/model/model.pt")
MODEL_CARD_PATH = Path("data/model/model.json")

LEGACY_ARCH = "gpt"


def save_checkpoint(
    path: Path, model: LanguageModel, arch: str, itos: list[str], meta: dict[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"arch": arch, "config": asdict(model.cfg), "model": model.state_dict(), "itos": itos, "meta": meta},
        path,
    )


def load_checkpoint(path: Path, backend: Backend) -> tuple[LanguageModel, dict[str, Any]]:
    """Return the model in eval mode on the backend's device, and the raw checkpoint dict."""
    if not path.exists():
        raise SystemExit(f"{path} not found - run `pdm run train` first")
    ckpt = torch.load(path, map_location=backend.device)
    cls = architecture(ckpt.get("arch", LEGACY_ARCH))
    model = backend.to(cls(cls.config_cls(**ckpt["config"])))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt
