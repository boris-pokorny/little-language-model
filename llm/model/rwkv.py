"""A tiny RWKV-4 language model (Peng et al., 2023).

Embedding -> LayerNorm -> pre-norm residual blocks -> LayerNorm -> head. Each
block is a time-mixing sub-layer followed by a channel-mixing sub-layer. Both
blend each token with the previous one ("token shift"). Time mixing computes,
per channel, an exponentially decayed weighted average of past values with
weights exp(k) (the WKV recurrence), gated by a sigmoid receptance. Channel
mixing is a squared-ReLU feed-forward layer with the same gate. There is no
positional embedding.

The WKV recurrence is a Python loop over time in the numerically stable
running-max form: O(T) per layer, state of 3 x d_model per sequence, plain
PyTorch, device-agnostic. Generation uses the inherited `generate`, which
re-feeds the last `block_size` tokens instead of carrying the state.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from llm.model.base import LanguageModel, ModelConfig
from llm.model.registry import register


@dataclass
class RWKVConfig(ModelConfig):
    """Hyperparameters of TinyRWKV.

    Attributes:
        ffn_mult: Channel-mix hidden width as a multiple of `d_model`.
    """

    ffn_mult: int = 4

    @property
    def d_ffn(self) -> int:
        """Channel-mix hidden width."""
        return self.ffn_mult * self.d_model


def token_shift(x: torch.Tensor) -> torch.Tensor:
    """Shifts a sequence one step forward in time.

    Args:
        x: Tensor of shape (B, T, C).

    Returns:
        Tensor of shape (B, T, C) holding x[:, t - 1] at position t and zeros at t = 0.
    """
    return F.pad(x, (0, 0, 1, -1))


def mix_ramp(cfg: RWKVConfig, layer_id: int) -> torch.Tensor:
    """Initial token-shift mix weight: channel i gets (i / C) ** (layer_id / n_layer).

    Args:
        cfg: Model configuration.
        layer_id: Zero-based block index.

    Returns:
        Float tensor of shape (1, 1, C).
    """
    ratio = layer_id / max(cfg.n_layer, 1)
    return (torch.arange(cfg.d_model, dtype=torch.float32) / cfg.d_model).pow(ratio).view(1, 1, -1)


class TimeMix(nn.Module):
    """Time-mixing sub-layer: token shift -> k, v, r -> WKV -> r * wkv -> output.

    Attributes:
        time_decay: Log decay per channel, shape (C,); the step decay is w = -exp(time_decay).
        time_first: Bonus added to the current token's key, shape (C,).
        time_mix_k: Token-shift mix weight of the key branch, shape (1, 1, C).
        time_mix_v: Token-shift mix weight of the value branch, shape (1, 1, C).
        time_mix_r: Token-shift mix weight of the receptance branch, shape (1, 1, C).
        key: Linear C -> C.
        value: Linear C -> C.
        receptance: Linear C -> C, followed by a sigmoid.
        output: Linear C -> C.
        dropout: Dropout on the sub-layer output.
    """

    def __init__(self, cfg: RWKVConfig, layer_id: int) -> None:
        """Initializes parameters with the RWKV-4 reference scheme.

        Args:
            cfg: Model configuration.
            layer_id: Zero-based block index; controls the layer-dependent init.
        """
        super().__init__()
        C = cfg.d_model
        # Decay rates spread per channel from fast to slow; steeper in deeper layers.
        ratio_0_to_1 = layer_id / max(cfg.n_layer - 1, 1)
        channel = torch.arange(C, dtype=torch.float32) / max(C - 1, 1)
        self.time_decay = nn.Parameter(-5 + 8 * channel.pow(0.7 + 1.3 * ratio_0_to_1))
        # Bonus of log(0.3) with a -0.5 / 0 / +0.5 offset cycling over channels.
        zigzag = 0.5 * (torch.arange(C) % 3 - 1)
        self.time_first = nn.Parameter(torch.full((C,), math.log(0.3)) + zigzag)
        ramp = mix_ramp(cfg, layer_id)
        self.time_mix_k = nn.Parameter(ramp.clone())
        self.time_mix_v = nn.Parameter(ramp + 0.3 * ratio_0_to_1)
        self.time_mix_r = nn.Parameter((0.5 * ramp).clone())
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.receptance = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies time mixing.

        Args:
            x: Normalized activations of shape (B, T, C).

        Returns:
            Residual update of shape (B, T, C).
        """
        xx = token_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k, v = self.key(xk), self.value(xv)
        r = torch.sigmoid(self.receptance(xr))
        return self.dropout(self.output(r * self.wkv(k, v)))

    def wkv(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Computes the WKV recurrence.

        Per channel, with w = -exp(time_decay) and u = time_first:

            y_t = (sum_{i<t} e^{(t-1-i) w + k_i} v_i + e^{u + k_t} v_t)
                / (sum_{i<t} e^{(t-1-i) w + k_i}     + e^{u + k_t})

        Numerator, denominator and their shared maximum exponent are carried as
        the state (a, b, p) so no exponential overflows.

        Args:
            k: Keys of shape (B, T, C).
            v: Values of shape (B, T, C).

        Returns:
            Tensor y of shape (B, T, C).
        """
        B, _, C = k.shape
        w = -torch.exp(self.time_decay)
        u = self.time_first
        a = k.new_zeros(B, C)
        b = k.new_zeros(B, C)
        p = k.new_full((B, C), float("-inf"))
        ys = []
        # unbind(1) instead of k[:, t]: autograd then stacks one gradient rather than
        # allocating a full-size zero tensor per timestep in the backward pass.
        for k_t, v_t in zip(k.unbind(1), v.unbind(1)):
            # Output at t includes the current token with bonus u.
            ww = u + k_t
            q = torch.maximum(p, ww)
            e1, e2 = torch.exp(p - q), torch.exp(ww - q)
            ys.append((e1 * a + e2 * v_t) / (e1 * b + e2))
            # State for t + 1: decay the history by w, add the current token without the bonus.
            ww = p + w
            q = torch.maximum(ww, k_t)
            e1, e2 = torch.exp(ww - q), torch.exp(k_t - q)
            a, b, p = e1 * a + e2 * v_t, e1 * b + e2, q
        return torch.stack(ys, dim=1)


class ChannelMix(nn.Module):
    """Channel-mixing sub-layer: token shift -> relu(key)^2 -> value, gated by sigmoid(receptance).

    Attributes:
        time_mix_k: Token-shift mix weight of the key branch, shape (1, 1, C).
        time_mix_r: Token-shift mix weight of the receptance branch, shape (1, 1, C).
        key: Linear C -> d_ffn.
        value: Linear d_ffn -> C.
        receptance: Linear C -> C, followed by a sigmoid.
        dropout: Dropout on the sub-layer output.
    """

    def __init__(self, cfg: RWKVConfig, layer_id: int) -> None:
        """Initializes parameters.

        Args:
            cfg: Model configuration.
            layer_id: Zero-based block index; controls the mix-weight init.
        """
        super().__init__()
        C = cfg.d_model
        ramp = mix_ramp(cfg, layer_id)
        self.time_mix_k = nn.Parameter(ramp.clone())
        self.time_mix_r = nn.Parameter(ramp.clone())
        self.key = nn.Linear(C, cfg.d_ffn, bias=False)
        self.value = nn.Linear(cfg.d_ffn, C, bias=False)
        self.receptance = nn.Linear(C, C, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies channel mixing.

        Args:
            x: Normalized activations of shape (B, T, C).

        Returns:
            Residual update of shape (B, T, C).
        """
        xx = token_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = torch.relu(self.key(xk)).square()
        return self.dropout(torch.sigmoid(self.receptance(xr)) * self.value(k))


class RWKVBlock(nn.Module):
    """Pre-norm residual block: time mixing, then channel mixing.

    Attributes:
        ln1: LayerNorm before `att`.
        att: `TimeMix` sub-layer.
        ln2: LayerNorm before `ffn`.
        ffn: `ChannelMix` sub-layer.
    """

    def __init__(self, cfg: RWKVConfig, layer_id: int) -> None:
        """Initializes the block.

        Args:
            cfg: Model configuration.
            layer_id: Zero-based block index.
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.att = TimeMix(cfg, layer_id)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = ChannelMix(cfg, layer_id)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies both sub-layers with residual connections.

        Args:
            x: Activations of shape (B, T, C).

        Returns:
            Tensor of shape (B, T, C).
        """
        x = x + self.att(self.ln1(x))
        return x + self.ffn(self.ln2(x))


@register("rwkv")
class TinyRWKV(LanguageModel):
    """RWKV-4 next-token language model, registered as "rwkv".

    Attributes:
        cfg: The `RWKVConfig` the model was built from.
        tok_emb: Token embedding.
        drop: Dropout on the embedding.
        ln0: LayerNorm on the embedding.
        blocks: Stack of `n_layer` `RWKVBlock`s.
        ln_f: Final LayerNorm.
        head: Linear d_model -> vocab_size.
    """

    config_cls = RWKVConfig

    def __init__(self, cfg: RWKVConfig) -> None:
        """Builds the model.

        Args:
            cfg: Model configuration.
        """
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.ln0 = nn.LayerNorm(cfg.d_model)
        self.blocks = nn.ModuleList(RWKVBlock(cfg, i) for i in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Computes next-token logits and, given targets, the loss.

        Args:
            idx: Token ids of shape (B, T), T <= `cfg.block_size`.
            targets: Next-token ids of shape (B, T), or None.

        Returns:
            Tuple (logits, loss): logits of shape (B, T, vocab_size); loss is the
            mean cross-entropy, or None when `targets` is None.

        Raises:
            AssertionError: If T > `cfg.block_size`.
        """
        _, T = idx.shape
        assert T <= self.cfg.block_size, "sequence longer than block_size"
        x = self.ln0(self.drop(self.tok_emb(idx)))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss
