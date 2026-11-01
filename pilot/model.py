"""Small GPT with RoPE and z-conditioned FiLM after every norm.

z=None   -> plain transformer (baseline).
z [B,zd] -> h = h * (1 + gamma(z)) + beta(z) after each RMSNorm (25 norms for 12 layers).
Fixed codes are rows of `codebook`; continuous z is sampled N(0, I) by the caller.
"""
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelCfg:
    vocab: int
    n_layer: int = 12
    n_head: int = 6
    d: int = 384
    max_len: int = 1024
    zdim: int = 16
    film_hidden: int = 64
    n_codes: int = 4
    film_init_std: float = 0.02

    def to_dict(self):
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.w, 1e-6)


def rope_cache(hd, max_len, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2).float() / hd))
    t = torch.arange(max_len).float()
    f = torch.outer(t, inv)
    return torch.cos(f), torch.sin(f)


def apply_rope(x, cos, sin):
    # x [B,H,T,hd]; cos/sin broadcastable to [B,1,T,hd/2]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


def film(x, gb, i):
    if gb is None:
        return x
    return x * (1 + gb[:, i, 0].unsqueeze(1)) + gb[:, i, 1].unsqueeze(1)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head, self.d = cfg.n_head, cfg.d
        self.norm1, self.norm2 = RMSNorm(cfg.d), RMSNorm(cfg.d)
        self.qkv = nn.Linear(cfg.d, 3 * cfg.d, bias=False)
        self.proj = nn.Linear(cfg.d, cfg.d, bias=False)
        self.fc = nn.Linear(cfg.d, 4 * cfg.d, bias=False)
        self.out = nn.Linear(4 * cfg.d, cfg.d, bias=False)

    def forward(self, x, cos, sin, gb, li, mask=None, cache=None, cache_pos=0):
        B, T, _ = x.shape
        h = film(self.norm1(x), gb, 2 * li)
        q, k, v = self.qkv(h).view(B, T, 3, self.n_head, self.d // self.n_head).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            kc, vc = cache
            kc[:, :, cache_pos:cache_pos + T] = k.to(kc.dtype)
            vc[:, :, cache_pos:cache_pos + T] = v.to(vc.dtype)
            k, v = kc[:, :, :cache_pos + T], vc[:, :, :cache_pos + T]
            q = q.to(k.dtype)
        if mask is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, self.d))
        h = film(self.norm2(x), gb, 2 * li + 1)
        return x + self.out(F.gelu(self.fc(h)))


class GPT(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab, cfg.d)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_f = RMSNorm(cfg.d)
        self.n_film = 2 * cfg.n_layer + 1
        self.film_net = nn.Sequential(nn.Linear(cfg.zdim, cfg.film_hidden), nn.GELU(),
                                      nn.Linear(cfg.film_hidden, self.n_film * 2 * cfg.d))
        self.codebook = nn.Parameter(torch.randn(cfg.n_codes, cfg.zdim))
        cos, sin = rope_cache(cfg.d // cfg.n_head, cfg.max_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        for b in self.blocks:
            nn.init.normal_(b.proj.weight, std=0.02 / math.sqrt(2 * cfg.n_layer))
            nn.init.normal_(b.out.weight, std=0.02 / math.sqrt(2 * cfg.n_layer))
        # small (not zero) so that modes differ slightly from the start and WTA can break ties
        nn.init.normal_(self.film_net[2].weight, std=cfg.film_init_std / math.sqrt(cfg.film_hidden))
        nn.init.zeros_(self.film_net[2].bias)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def n_params(self, trunk_only=True):
        skip = ("film_net", "codebook") if trunk_only else ()
        return sum(p.numel() for n, p in self.named_parameters() if not n.startswith(skip))

    def code_z(self, codes):
        return self.codebook[codes]

    def _gb(self, z):
        if z is None:
            return None
        return self.film_net(z.to(self.film_net[0].weight.dtype)).view(z.shape[0], self.n_film, 2, self.cfg.d)

    def _logits(self, x, gb):
        x = film(self.norm_f(x), gb, self.n_film - 1)
        return F.linear(x, self.emb.weight)

    def forward(self, idx, z=None):
        """Training / scoring forward on right-padded batches (causal)."""
        T = idx.shape[1]
        gb = self._gb(z)
        cos, sin = self.cos[:T].view(1, 1, T, -1), self.sin[:T].view(1, 1, T, -1)
        x = self.emb(idx)
        for li, b in enumerate(self.blocks):
            x = b(x, cos, sin, gb, li)
        return self._logits(x, gb)

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(self, prompts, z=None, temperature=0.0, max_len=None, eos_id=2, pad_id=0,
                 batch_size=512, seed=0, max_new=None):
        """Sample continuations. prompts: list[list[int]]; z: [N,zdim] tensor or None.

        Returns list of generated id lists (EOS stripped)."""
        max_len = max_len or self.cfg.max_len
        dev = self.emb.weight.device
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        outs = [None] * len(prompts)
        gen = torch.Generator(device=dev)
        gen.manual_seed(seed)
        for s in range(0, len(order), batch_size):
            idxs = order[s:s + batch_size]
            zb = None if z is None else z[idxs].to(dev)
            with autocast(dev):
                res = self._generate_batch([prompts[i] for i in idxs], zb, temperature, max_len, eos_id,
                                           pad_id, gen, max_new)
            for i, r in zip(idxs, res):
                outs[i] = r
        return outs

    def _generate_batch(self, prompts, z, temperature, max_len, eos_id, pad_id, gen, max_new):
        dev = self.emb.weight.device
        B, Lp = len(prompts), max(len(p) for p in prompts)
        budget = max_len - Lp
        if max_new is not None:
            budget = min(budget, max_new)
        if budget <= 0:
            return [[] for _ in prompts]
        L = Lp + budget
        ids = torch.full((B, Lp), pad_id, dtype=torch.long, device=dev)
        valid = torch.zeros((B, L), dtype=torch.bool, device=dev)
        for i, p in enumerate(prompts):
            ids[i, Lp - len(p):] = torch.tensor(p, device=dev)
            valid[i, Lp - len(p):] = True
        valid[:, Lp:] = True
        pos = (valid[:, :Lp].long().cumsum(1) - 1).clamp(min=0)
        dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
        hd = self.cfg.d // self.cfg.n_head
        caches = [(torch.empty(B, self.cfg.n_head, L, hd, device=dev, dtype=dtype),
                   torch.empty(B, self.cfg.n_head, L, hd, device=dev, dtype=dtype)) for _ in self.blocks]
        gb = self._gb(z)
        causal = torch.tril(torch.ones(Lp, Lp, dtype=torch.bool, device=dev))
        eye = torch.eye(Lp, dtype=torch.bool, device=dev)
        mask = ((causal & valid[:, None, :Lp]) | eye)[:, None]          # pads attend to themselves only
        logits = self._step(ids, pos, mask, caches, 0, gb)[:, -1]
        cur_pos = pos[:, -1] + 1
        out = torch.full((B, budget), eos_id, dtype=torch.long, device=dev)
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        for t in range(budget):
            tok = self._sample(logits, temperature, gen)
            tok = torch.where(done, torch.full_like(tok, eos_id), tok)
            out[:, t] = tok
            done |= tok == eos_id
            if bool(done.all()) or t == budget - 1:
                break
            m = valid[:, None, None, :Lp + t + 1]
            logits = self._step(tok[:, None], cur_pos[:, None], m, caches, Lp + t, gb)[:, -1]
            cur_pos = cur_pos + 1
        res = []
        for row in out.tolist():
            res.append(row[:row.index(eos_id)] if eos_id in row else row)
        return res

    def _step(self, idx, pos, mask, caches, cache_pos, gb):
        cos = self.cos[pos].unsqueeze(1)
        sin = self.sin[pos].unsqueeze(1)
        x = self.emb(idx)
        for li, b in enumerate(self.blocks):
            x = b(x, cos, sin, gb, li, mask=mask, cache=caches[li], cache_pos=cache_pos)
        return self._logits(x, gb).float()

    @staticmethod
    def _sample(logits, temperature, gen):
        if temperature <= 0:
            return logits.argmax(-1)
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, 1, generator=gen).squeeze(1)


def per_example_nll(logits, targets):
    """Mean NLL over non-ignored targets of each example -> [B] (float32)."""
    B, T, V = logits.shape
    nll = F.cross_entropy(logits.float().reshape(-1, V), targets.reshape(-1), ignore_index=-100,
                          reduction="none").view(B, T)
    cnt = (targets != -100).sum(1).clamp(min=1)
    return nll.sum(1) / cnt


@torch.no_grad()
def score_positions(model, seqs, positions, z=None, batch_size=256):
    """Next-token log-probs at given positions. seqs: list[list[int]], positions: list[list[int]].

    Returns list of [len(positions[i]), V] float tensors on CPU."""
    dev = model.emb.weight.device
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    out = [None] * len(seqs)
    for s in range(0, len(order), batch_size):
        idxs = order[s:s + batch_size]
        T = max(len(seqs[i]) for i in idxs)
        ids = torch.zeros((len(idxs), T), dtype=torch.long, device=dev)
        for r, i in enumerate(idxs):
            ids[r, :len(seqs[i])] = torch.tensor(seqs[i], device=dev)
        zb = None if z is None else z[idxs].to(dev)
        with autocast(dev):
            logits = model(ids, zb)
        lp = torch.log_softmax(logits.float(), -1)
        for r, i in enumerate(idxs):
            out[i] = lp[r, positions[i]].cpu()
    return out


def autocast(dev):
    if dev.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)
