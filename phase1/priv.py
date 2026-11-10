"""Shared + private LoRA per mode (+ FiLM codes): the sweep from wta1 (all shared) to LoRA-MCL (all private).

Every LoRA target of mode k computes
    W x + s * (B_s A_s x + B_k A_k x),     s = alpha / (r_s + r_p),
a rank r_s + r_p adapter whose first r_s directions (A_s, B_s) every mode shares and whose last r_p (A_k, B_k) belong
to mode k only. r_p = 0 is the LoRA of wta1, r_s = 0 the K separate adapters of LoRA-MCL; at a fixed per-mode rank
R = r_s + r_p the private share r_p / R moves between the two with the same scale and the same cost per example.
FiLM codes (phase1/film.py: FiLM, FiLMNorm) scale the norms on top, unchanged. Without modes (K = 1, r_p = 0) it is
the plain LoRA of SFT.

Export: mode k folds into a plain checkpoint (LoRA merged, FiLM code k folded into the norm weights), as
film.export_modes. Adapters of phase1.train (film.py format: wta1, sft1_eq, loramcl) load through from_film_format."""
import math
import os
import shutil

import torch
import torch.nn as nn

from phase1 import film as FM

LORA_TARGETS = FM.LORA_TARGETS
SUFFIXES = (".A_s", ".B_s", ".A_p", ".B_p")


class PrivLoRALinear(nn.Module):
    def __init__(self, base, r_s, r_p, K, alpha, ctx, dropout=0.0):
        super().__init__()
        self.base, self.ctx, self.r_s, self.r_p = base, ctx, r_s, r_p
        self.scale = alpha / (r_s + r_p)
        i, o = base.in_features, base.out_features
        if r_s:
            self.A_s = nn.Parameter(torch.randn(r_s, i) / math.sqrt(i))
            self.B_s = nn.Parameter(torch.zeros(o, r_s))
        if r_p:
            self.A_p = nn.Parameter(torch.randn(K, r_p, i) / math.sqrt(i))
            self.B_p = nn.Parameter(torch.zeros(K, o, r_p))
        self.drop = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        out = self.base(x)
        h = self.drop(x) if self.drop is not None else x
        if self.r_s:
            out = out + (h @ self.A_s.t().to(h.dtype)) @ self.B_s.t().to(h.dtype) * self.scale
        if self.r_p:
            m = self.ctx.modes
            if m is None:
                raise RuntimeError("private LoRA needs ctx.modes")
            if m.numel() == 1:                  # one mode for every row (generation, beam search)
                k = int(m.reshape(-1)[0])
                A, B = self.A_p[k].to(h.dtype), self.B_p[k].to(h.dtype)
                out = out + (h @ A.t()) @ B.t() * self.scale
            else:                               # mode per row: [B, T, in] x [B, in, r]
                A, B = self.A_p[m].to(h.dtype), self.B_p[m].to(h.dtype)
                out = out + (h @ A.transpose(1, 2)) @ B.transpose(1, 2) * self.scale
        return out

    def delta(self, k):
        """Weight update of mode k in float32: s * (B_s A_s + B_k A_k)."""
        d = torch.zeros(self.base.out_features, self.base.in_features)
        if self.r_s:
            d += self.B_s.detach().float().cpu() @ self.A_s.detach().float().cpu()
        if self.r_p:
            d += self.B_p[k].detach().float().cpu() @ self.A_p[k].detach().float().cpu()
        return d * self.scale


def lora_modules(model):
    """(key, parent, name) of every LoRA target, in a fixed order."""
    out = []
    for li, layer in enumerate(model.model.layers):
        for pname, parent in (("self_attn", layer.self_attn), ("mlp", layer.mlp)):
            for name in LORA_TARGETS:
                if hasattr(parent, name):
                    out.append((f"model.layers.{li}.{pname}.{name}", parent, name))
    return out


def build(base_path, K=1, r_s=32, r_p=0, alpha=64.0, film=False, device="cuda", grad_ckpt=True, dropout=0.0,
          dtype=None):
    """Frozen base + shared/private LoRA (+ FiLM codes). Returns (model, film or None, ctx)."""
    from transformers import AutoModelForCausalLM
    if r_s + r_p < 1:
        raise ValueError("r_s + r_p must be >= 1")
    if r_p and K < 2:
        raise ValueError("a private LoRA needs K >= 2 modes")
    if dtype is None:
        dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base_path, dtype=dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    ctx = FM.Ctx()
    for key, parent, name in lora_modules(model):
        setattr(parent, name, PrivLoRALinear(getattr(parent, name), r_s, r_p, K, alpha, ctx, dropout))
    fl = None
    if film:
        mods = FM.norm_modules(model)
        for i, (parent, name) in enumerate(mods):
            setattr(parent, name, FM.FiLMNorm(getattr(parent, name), i, ctx))
        fl = FM.FiLM(len(mods), model.config.hidden_size, K=K)
    model.config.use_cache = False
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device)
    if fl is not None:
        fl.to(device)
    return model, fl, ctx


def build_from(base_path, ck, device="cuda", grad_ckpt=False, dtype=None):
    """Model of a saved adapter (priv format or film.py format), weights loaded."""
    ck = as_priv(ck)
    c = ck["cfg"]
    model, fl, ctx = build(base_path, c["K"], c["r_s"], c["r_p"], c["alpha"], c["film"], device, grad_ckpt,
                           0.0, dtype)
    load_state(model, fl, ck["adapter"])
    return model, fl, ctx, c


def trainable(model, film):
    ps = [p for n, p in model.named_parameters() if n.endswith(SUFFIXES)]
    for p in ps:
        p.data = p.data.float()
        p.requires_grad_(True)
    return ps, (list(film.parameters()) if film is not None else [])


def adapter_state(model, film):
    st = {n: p.detach().cpu() for n, p in model.named_parameters() if n.endswith(SUFFIXES)}
    if film is not None:
        st.update({f"film.{n}": p.detach().cpu() for n, p in film.named_parameters()})
    return st


def load_state(model, film, st):
    params = dict(model.named_parameters())
    fparams = dict(film.named_parameters()) if film is not None else {}
    want = {n for n in params if n.endswith(SUFFIXES)} | {f"film.{n}" for n in fparams}
    if set(st) != want:
        raise ValueError(f"adapter keys differ: missing {sorted(want - set(st))[:3]}, extra {sorted(set(st) - want)[:3]}")
    for n, v in st.items():
        (fparams[n[5:]] if n.startswith("film.") else params[n]).data.copy_(v)


def n_params(cfg, model_cfg):
    """Trainable parameters (LoRA shared, LoRA private, FiLM) of a configuration on a base config."""
    d, L, f = model_cfg.hidden_size, model_cfg.num_hidden_layers, model_cfg.intermediate_size
    hd = getattr(model_cfg, "head_dim", None) or d // model_cfg.num_attention_heads
    q, kv = model_cfg.num_attention_heads * hd, model_cfg.num_key_value_heads * hd
    per_rank = L * ((d + q) + 2 * (d + kv) + (q + d) + 2 * (d + f) + (f + d))
    n_norm = 2 * L + 1
    film = (cfg["K"] * 16 + 16 * 64 + 64 + 64 * n_norm * d + n_norm * d) if cfg["film"] else 0
    return {"shared": per_rank * cfg["r_s"], "private": per_rank * cfg["r_p"] * cfg["K"], "film": film}


def as_priv(ck):
    """A checkpoint in priv format; adapters of phase1.train (film.py format) are converted (exactly the same
    function: the scale alpha / r of film.py equals alpha / (r_s + r_p))."""
    c = ck["cfg"]
    if c.get("format") == "priv":
        return ck
    if c.get("prefix_modes") or c.get("film_z"):
        raise ValueError("ModC prefixes and continuous z are not supported")
    multi = c.get("lora_modes", 0)
    K = c.get("film_modes", 0) or multi or 1
    st = {}
    for n, v in ck["adapter"].items():
        if n.startswith("film."):
            st[n] = v
        elif n.endswith(".A") or n.endswith(".B"):
            st[n + ("_p" if multi else "_s")] = v
        else:
            raise ValueError(f"unknown adapter key {n}")
    cfg = {"format": "priv", "K": K, "r_s": 0 if multi else c["lora_r"], "r_p": c["lora_r"] if multi else 0,
           "alpha": float(c["lora_alpha"]), "film": bool(c.get("film_modes")), "variant": c.get("variant", "?"),
           "converted": True}
    return {"adapter": st, "cfg": cfg}


def load_ckpt(path):
    return as_priv(torch.load(path, map_location="cpu", weights_only=False))


@torch.no_grad()
def export_mode(base_path, ck, out, mode, dtype=torch.bfloat16):
    """Plain checkpoint of mode `mode` (None without modes): LoRA merged, FiLM code folded into the norms. CPU."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ck = as_priv(ck)
    c, st = ck["cfg"], ck["adapter"]
    if c["r_p"] and mode is None:
        raise ValueError("a model with private LoRA needs a mode")
    model = AutoModelForCausalLM.from_pretrained(base_path, dtype=dtype)
    s = c["alpha"] / (c["r_s"] + c["r_p"])
    for key, parent, name in lora_modules(model):
        lin = getattr(parent, name)
        d = torch.zeros(lin.out_features, lin.in_features)
        if c["r_s"]:
            d += st[f"{key}.B_s"].float() @ st[f"{key}.A_s"].float()
        if c["r_p"]:
            d += st[f"{key}.B_p"][mode].float() @ st[f"{key}.A_p"][mode].float()
        lin.weight.copy_((lin.weight.float() + s * d).to(lin.weight.dtype))
    if c["film"] and mode is not None:
        norms = [getattr(p, n) for p, n in FM.norm_modules(model)]
        fl = FM.FiLM(len(norms), model.config.hidden_size, K=c["K"])
        fl.load_state_dict({k[5:]: v for k, v in st.items() if k.startswith("film.")})
        g = fl.gamma(torch.tensor([mode]))[0].float()
        for nm, gl in zip(norms, g):
            nm.weight.copy_((nm.weight.float() * (1 + gl)).to(nm.weight.dtype))
    if os.path.exists(out):
        shutil.rmtree(out)
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(base_path).save_pretrained(out)
    return out


def set_mode(ctx, film, mode, n=1, device="cpu"):
    """Every row of the next forward passes uses mode `mode` (None: no modes)."""
    if mode is None:
        ctx.modes, ctx.gamma = None, None
        return
    ctx.modes = torch.tensor([mode], device=device)
    ctx.gamma = film.gamma(ctx.modes) if film is not None else None
