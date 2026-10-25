"""Contract every architecture honours, and the sampling loop they share.

Callers (trainer, evaluator, chat, agent) depend only on `LanguageModel`:
`cfg.block_size`, `forward(idx, targets) -> (logits, loss)` and `generate`.
Architecture-specific knobs live in each architecture's own config subclass.

Device-agnostic: nothing here knows about CUDA/CPU.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, ClassVar

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelConfig:
    """Fields every architecture shares; subclasses add their own."""

    vocab_size: int
    block_size: int = 64
    n_layer: int = 4
    d_model: int = 128
    dropout: float = 0.0


class LanguageModel(nn.Module, ABC):
    """A next-token language model over a context of at most `cfg.block_size` tokens."""

    config_cls: ClassVar[type[ModelConfig]] = ModelConfig
    cfg: ModelConfig

    @abstractmethod
    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return logits of shape (B, T, vocab) and, given targets, the mean cross-entropy."""

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        logits_filter: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Sample continuations. `logits_filter(idx, logits)` may mask logits
        (set to -inf) given the sequence so far, e.g. top-k or a lexicon."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size :])
            logits = logits[:, -1, :] / temperature
            if logits_filter is not None:
                logits = logits_filter(idx, logits)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
