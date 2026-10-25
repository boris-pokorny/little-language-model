"""A tiny selective state-space language model (after Gu & Dao, 2023).

Same skeleton as TinyGPT (embedding -> pre-norm residual blocks -> LayerNorm
-> head) with the attention + MLP pair replaced by one selective-SSM mixer per
block: a gated branch with a short depthwise causal convolution feeding a
diagonal linear recurrence whose step size, input and output projections are
functions of the token (that is what "selective" means). No positional
embedding: the recurrence knows the order on its own.

The scan is an explicit Python loop over time, O(T) per layer with a fixed
state of d_inner x d_state per sequence. Plain PyTorch, device-agnostic; the
fused kernels of the reference implementation are not needed at block_size 64.
Generation re-feeds the last block_size tokens like TinyGPT rather than caching
the state, so both architectures go through the same `generate`.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from llm.model.base import LanguageModel, ModelConfig
from llm.model.registry import register


@dataclass
class SSMConfig(ModelConfig):
    d_state: int = 16  # size of the recurrent state per channel
    d_conv: int = 4  # width of the causal depthwise convolution
    expand: int = 2  # d_inner = expand * d_model

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model

    @property
    def dt_rank(self) -> int:
        return math.ceil(self.d_model / 16)


class SelectiveSSM(nn.Module):
    """The mixer: in_proj -> conv -> selective scan -> gate -> out_proj."""

    def __init__(self, cfg: SSMConfig) -> None:
        super().__init__()
        d_inner, d_state = cfg.d_inner, cfg.d_state
        self.in_proj = nn.Linear(cfg.d_model, 2 * d_inner, bias=False)
        self.conv = nn.Conv1d(
            d_inner, d_inner, cfg.d_conv, groups=d_inner, padding=cfg.d_conv - 1
        )
        self.x_proj = nn.Linear(d_inner, cfg.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(cfg.dt_rank, d_inner)
        # A is negative real and diagonal: state h decays by exp(dt * A) each step.
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(d_inner, 1))
        self.D = nn.Parameter(torch.ones(d_inner))  # skip connection per channel
        self.out_proj = nn.Linear(d_inner, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.d_state, self.dt_rank = d_state, cfg.dt_rank

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        _, T, _ = u.shape
        x, z = self.in_proj(u).chunk(2, dim=-1)  # (B, T, d_inner) each
        x = self.conv(x.transpose(1, 2))[:, :, :T].transpose(1, 2)  # causal: keep the first T
        x = F.silu(x)
        y = self.scan(x)
        return self.dropout(self.out_proj(y * F.silu(z)))

    def scan(self, x: torch.Tensor) -> torch.Tensor:
        """Selective scan over time: h_t = exp(dt_t A) h_{t-1} + dt_t B_t x_t;  y_t = C_t h_t + D x_t."""
        B_, T, d_inner = x.shape
        dt, Bm, Cm = torch.split(self.x_proj(x), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # (B, T, d_inner), positive step sizes
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        decay = torch.exp(dt.unsqueeze(-1) * A)  # (B, T, d_inner, d_state)
        drive = (dt * x).unsqueeze(-1) * Bm.unsqueeze(2)  # (B, T, d_inner, d_state)
        # unbind (not decay[:, t]) so autograd stacks one gradient instead of
        # materialising a full-size zero tensor per timestep in the backward pass
        h = x.new_zeros(B_, d_inner, self.d_state)
        ys = []
        for decay_t, drive_t, C_t in zip(decay.unbind(1), drive.unbind(1), Cm.unbind(1)):
            h = decay_t * h + drive_t
            ys.append((h * C_t.unsqueeze(1)).sum(-1))  # (B, d_inner)
        return torch.stack(ys, dim=1) + x * self.D


class SSMBlock(nn.Module):
    def __init__(self, cfg: SSMConfig) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(cfg.d_model)
        self.mixer = SelectiveSSM(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.ln(x))


@register("ssm")
class TinySSM(LanguageModel):
    config_cls = SSMConfig

    def __init__(self, cfg: SSMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(SSMBlock(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, T = idx.shape
        assert T <= self.cfg.block_size, "sequence longer than block_size"
        x = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss
