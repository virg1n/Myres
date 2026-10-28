"""Decoder-only transformer with a causal *writer* stream and a bidirectional
*re-read* stream over the last W tokens (the window).

One forward pass processes a packed sequence

    [ causal stream: prompt + solution (positions 0..T-1) ]
    [ window copy 1 (positions s1..t1-1) ] [ window copy 2 ] ...

* causal tokens attend causally to causal tokens (a standard LM);
* a window copy attends to the causal prefix before its window start, and
  bidirectionally to the tokens of its own copy ("reread the window using
  context on both sides").

Heads:
* lm_head  (causal stream): next-token prediction = "write".
* err_head (window copy): is this token a (root) error?
* fix_head (window copy): what token should be here? (fix / fill a mask).
* dep_q / dep_k (window copy): bilinear pairwise head; dep[j, k] = probability
  that token k depends on token j (predicts which text relied on an error).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 512
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_ff: int = 1024
    dep_dim: int = 64
    max_pos: int = 2048
    rope_base: float = 10000.0

    @classmethod
    def from_dict(cls, d, vocab_size):
        c = cls(vocab_size=vocab_size)
        for k, v in (d or {}).items():
            if hasattr(c, k):
                setattr(c, k, v)
        c.vocab_size = vocab_size
        return c


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


def rope_cache(max_pos, dim, base):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_pos).float()
    f = torch.outer(t, inv)
    return torch.cos(f), torch.sin(f)


def apply_rope(x, cos, sin):
    # x: (B, H, L, D); cos/sin: (B, 1, L, D/2)
    x1, x2 = x[..., ::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2).type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.h = cfg.n_heads
        self.dh = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, mask, cache=None, layer=0):
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            k, v = cache.update(layer, k, v)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(y.transpose(1, 2).reshape(B, L, D))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.n2 = RMSNorm(cfg.d_model)
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, mask, cache=None, layer=0):
        x = x + self.attn(self.n1(x), cos, sin, mask, cache, layer)
        h = self.n2(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


class RevisionLM(nn.Module):
    REVISION_PARAMS = ("stream_emb", "err_head", "fix_head", "dep_q", "dep_k")

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.stream_emb = nn.Embedding(2, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.err_head = nn.Linear(cfg.d_model, 1)
        self.fix_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.dep_q = nn.Linear(cfg.d_model, cfg.dep_dim)
        self.dep_k = nn.Linear(cfg.d_model, cfg.dep_dim)
        cos, sin = rope_cache(cfg.max_pos, cfg.d_model // cfg.n_heads, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        for b in self.blocks:
            nn.init.normal_(b.attn.o.weight, std=0.02 / math.sqrt(2 * cfg.n_layers))
            nn.init.normal_(b.w2.weight, std=0.02 / math.sqrt(2 * cfg.n_layers))
        nn.init.zeros_(self.stream_emb.weight)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def revision_parameters(self):
        return [p for n, p in self.named_parameters() if n.split(".")[0] in self.REVISION_PARAMS]

    def n_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def forward(self, ids, pos, stream, mask, cache=None):
        """ids/pos/stream: (B, L) long; mask: (B, 1, L, L_cache + L) bool (True =
        attend). With a KVCache, keys/values of these tokens are appended to it.
        Returns the final hidden states (B, L, d)."""
        x = self.tok_emb(ids) + self.stream_emb(stream)
        cos = self.rope_cos[pos].unsqueeze(1).to(x.dtype)
        sin = self.rope_sin[pos].unsqueeze(1).to(x.dtype)
        for i, blk in enumerate(self.blocks):
            x = blk(x, cos, sin, mask, cache, i)
        if cache is not None:
            cache.length += ids.shape[1]
        return self.norm(x)

    # ---- heads -------------------------------------------------------------
    def lm_logits(self, h):
        return F.linear(h, self.tok_emb.weight)

    def fix_logits(self, h):
        return self.fix_head(h)

    def err_logits(self, h):
        return self.err_head(h).squeeze(-1)

    def dep_logits(self, hq, hk):
        """hq: (..., d) rows, hk: (..., K, d) keys -> (..., K)."""
        q = self.dep_q(hq)
        k = self.dep_k(hk)
        return torch.einsum("...d,...kd->...k", q, k) / math.sqrt(q.shape[-1])


class KVCache:
    """Preallocated key/value cache for causal generation."""

    def __init__(self, cfg, batch, max_len, device, dtype):
        dh = cfg.d_model // cfg.n_heads
        self.k = [torch.zeros(batch, cfg.n_heads, max_len, dh, device=device, dtype=dtype) for _ in range(cfg.n_layers)]
        self.v = [torch.zeros(batch, cfg.n_heads, max_len, dh, device=device, dtype=dtype) for _ in range(cfg.n_layers)]
        self.length = 0

    def update(self, layer, k, v):
        n = k.shape[2]
        self.k[layer][:, :, self.length:self.length + n] = k
        self.v[layer][:, :, self.length:self.length + n] = v
        return self.k[layer][:, :, :self.length + n], self.v[layer][:, :, :self.length + n]


def build_mask(pos, is_copy, wstart, wid, valid):
    """Packed-sequence attention mask, see module docstring. All inputs (B, L).
    pos: RoPE positions; is_copy: bool; wstart: window start (causal position)
    for copy tokens; wid: window id for copy tokens; valid: non-pad.
    Returns (B, 1, L, L) bool."""
    pi, pj = pos.unsqueeze(2), pos.unsqueeze(1)
    ci, cj = is_copy.unsqueeze(2), is_copy.unsqueeze(1)
    causal = ~ci & ~cj & (pj <= pi)
    to_prefix = ci & ~cj & (pj < wstart.unsqueeze(2))
    same_win = ci & cj & (wid.unsqueeze(2) == wid.unsqueeze(1))
    m = (causal | to_prefix | same_win) & valid.unsqueeze(2) & valid.unsqueeze(1)
    eye = torch.eye(pos.shape[1], dtype=torch.bool, device=pos.device).unsqueeze(0)
    m = m | (eye & ~valid.unsqueeze(2))  # pad rows attend to themselves (avoid NaN)
    return m.unsqueeze(1)
