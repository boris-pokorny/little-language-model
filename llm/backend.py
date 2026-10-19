"""Hardware seam: the only module that knows about CUDA / MPS / CPU.

Model and training code never touch devices directly; they receive a
Backend and use backend.to(...) / backend.autocast(). Supporting another
accelerator API later means editing only this file.
"""

import contextlib
import logging
from dataclasses import dataclass
from typing import ContextManager, TypeVar

import torch

log = logging.getLogger(__name__)

T = TypeVar("T", torch.Tensor, torch.nn.Module)


@dataclass(frozen=True)
class Backend:
    device: torch.device

    def to(self, obj: T) -> T:
        return obj.to(self.device)

    def autocast(self) -> ContextManager[None]:
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


def _available(kind: str) -> bool:
    if kind == "cuda":
        return torch.cuda.is_available()
    if kind == "mps":
        return torch.backends.mps.is_available()
    return kind == "cpu"


def select_backend(name: str | None = None) -> Backend:
    """Pick a backend by name ("cpu", "cuda", "cuda:1", "mps"), or auto-detect."""
    if name is None:
        name = next(kind for kind in ("cuda", "mps", "cpu") if _available(kind))
    device = torch.device(name)
    if not _available(device.type):
        raise RuntimeError(f"{name} requested but not available on this machine")
    log.info(f"backend: {device}")
    return Backend(device)
