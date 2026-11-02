"""dlm-mini: a minimal Mixture-of-Depths (MoD) language model.

Dynamic compute for tokens: at MoD layers, a learned router sends only the
top-k tokens through the heavy sub-block; the rest skip it via the residual
path. This is the core idea from:

    "Mixture-of-Depths: Dynamically allocating compute in transformer-based
     language models" (Raposo et al., 2024, arXiv:2404.02258)

The implementation here also makes the FLOP savings real: skipped tokens are
never gathered into the sub-block. Most reference MoD code computes every
token and masks the output -- that is memory-equivalent, not compute-saving.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x


class RoPE(nn.Module):
    """Rotary position embeddings, cached to max_len."""

    def __init__(self, head_dim: int, max_len: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """x: (B, H, S, Dh); pos: (B, S) integer positions."""
        B, H, S, Dh = x.shape
        cos = self.cos_cached[pos].view(B, 1, S, Dh // 2)
        sin = self.sin_cached[pos].view(B, 1, S, Dh // 2)
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    """Causal self-attention with GQA.

    Note: for MoD layers the tokens handed in are a *subset* of the sequence,
    so causality is enforced within that subsequence (an approximation shared
    by most practical MoD implementations).
    """

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, max_len: int, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads

        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.rope = RoPE(self.head_dim, max_len)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q, pos)
        k = self.rope(k, pos)

        rep = self.n_heads // self.n_kv_heads
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).reshape(B, S, -1)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_mult: float = 8 / 3, dropout: float = 0.0, multiple_of: int = 64):
        super().__init__()
        hidden = int(dim * hidden_mult)
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class DenseSubBlock(nn.Module):
    """The heavy computational unit. A dense layer *is* one of these."""

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, max_len: int, dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, n_heads, n_kv_heads, max_len, dropout)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, dropout=dropout)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), pos)
        x = x + self.mlp(self.norm2(x))
        return x


class DenseBlock(nn.Module):
    """Standard transformer block: every token is computed."""

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, max_len: int, dropout: float):
        super().__init__()
        self.sub = DenseSubBlock(dim, n_heads, n_kv_heads, max_len, dropout)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sub(x, pos), x.new_zeros(())


class MoDBlock(nn.Module):
    """Mixture-of-Depths block.

    1. A per-token router scores every token.
    2. Only the top-k tokens (capacity ratio) are gathered and computed.
    3. Results are scattered back; skipped tokens keep their residual state.

    Gradients still reach the router through a straight-through sigmoid gate:
    forward masks hard, backward flows through the soft score.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        max_len: int,
        dropout: float,
        capacity: float = 0.5,
    ):
        super().__init__()
        assert 0.0 < capacity <= 1.0, "capacity must be in (0, 1]"
        self.sub = DenseSubBlock(dim, n_heads, n_kv_heads, max_len, dropout)
        self.router = nn.Linear(dim, 1, bias=False)
        self.capacity = capacity

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        scores = self.router(x).squeeze(-1)  # (B, T)

        k = max(1, int(round(T * self.capacity)))
        idx = scores.topk(k, dim=-1).indices  # (B, k)

        # straight-through binary gate: forward hard mask, backward soft prob
        soft = torch.sigmoid(scores)
        hard = torch.zeros_like(soft).scatter(1, idx, 1.0)
        gate = hard + soft - soft.detach()  # (B, T)

        # gather selected tokens -> compute -> scatter back (real FLOP savings)
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, k)
        sel_x = x[batch_idx, idx]                 # (B, k, D)
        sel_pos = pos[batch_idx, idx]             # (B, k)
        sel_out = self.sub(sel_x, sel_pos)        # (B, k, D)

        out = torch.zeros_like(x).scatter(
            1, idx.unsqueeze(-1).expand(-1, -1, D), sel_out
        )
        x = x + gate.unsqueeze(-1) * out

        # load-balancing auxiliary loss: nudge the router's soft probability
        # mass toward the capacity so it neither saturates nor starves
        aux = (soft.mean() - self.capacity) ** 2
        return x, aux


@dataclass
class DLMConfig:
    vocab_size: int = 65
    dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 2
    max_len: int = 512
    dropout: float = 0.0
    capacity: float = 0.5
    mod_every: int = 2      # every n-th layer is MoD; 0 disables MoD entirely
    aux_weight: float = 0.01


class DLM(nn.Module):
    """Small decoder-only LM with optional Mixture-of-Depths layers."""

    def __init__(self, cfg: DLMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)

        layers: list[nn.Module] = []
        for i in range(cfg.n_layers):
            if cfg.mod_every and i % cfg.mod_every == 0 and i > 0:
                layers.append(
                    MoDBlock(cfg.dim, cfg.n_heads, cfg.n_kv_heads, cfg.max_len, cfg.dropout, cfg.capacity)
                )
            else:
                layers.append(DenseBlock(cfg.dim, cfg.n_heads, cfg.n_kv_heads, cfg.max_len, cfg.dropout))
        self.layers = nn.ModuleList(layers)

        self.norm_f = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init_weights)
        # scaled init for residual projections (GPT-2 style)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        if T > self.cfg.max_len:
            raise ValueError(f"sequence length {T} exceeds max_len {self.cfg.max_len}")

        pos = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)
        x = self.drop(self.tok_emb(idx))

        aux_total = x.new_zeros(())
        for layer in self.layers:
            x, aux = layer(x, pos)
            aux_total = aux_total + aux

        x = self.norm_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
            loss = ce + self.cfg.aux_weight * aux_total
        return logits, loss, aux_total

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.max_len:]
            logits, _, _ = self(ctx)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, nxt), dim=1)
        return idx
