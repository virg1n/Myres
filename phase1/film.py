"""LoRA + scale-only FiLM on a Hugging Face Qwen3 model.

FiLM: after every decoder RMSNorm (input_layernorm, post_attention_layernorm) and the final norm,
    y = norm(x) * (1 + gamma_l(z)),   gamma = MLP(z),  z = one of K learned codes.
Because RMSNorm is weight * normalized(x), a fixed code folds exactly into the norm weights:
    weight_l <- weight_l * (1 + gamma_l(z_k)),
so every mode exports as a plain Qwen3 checkpoint (LoRA merged too) that vLLM can serve.
"""
import math
import os
import shutil

import torch
import torch.nn as nn

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.t().to(x.dtype)) @ self.B.t().to(x.dtype) * self.scale

    def merged_weight(self):
        return (self.base.weight.float() + self.scale * (self.B.float() @ self.A.float())).to(self.base.weight.dtype)


class MultiLoRALinear(nn.Module):
    """K independent adapters (LoRA-MCL); ctx.modes [B] picks the adapter of every batch row."""

    def __init__(self, base, r, alpha, K, ctx):
        super().__init__()
        self.base, self.ctx = base, ctx
        self.A = nn.Parameter(torch.randn(K, r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(K, base.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        m = self.ctx.modes
        A, B = self.A[m].to(x.dtype), self.B[m].to(x.dtype)
        return self.base(x) + (x @ A.transpose(1, 2)) @ B.transpose(1, 2) * self.scale


class Ctx:
    gamma = None          # [B, n_norms, d] or None
    modes = None          # [B] adapter index per row (MultiLoRALinear)


class FiLMNorm(nn.Module):
    def __init__(self, norm, idx, ctx):
        super().__init__()
        self.norm, self.idx, self.ctx = norm, idx, ctx

    def forward(self, x):
        y = self.norm(x)
        g = self.ctx.gamma
        if g is None:
            return y
        return y * (1 + g[:, self.idx].unsqueeze(1).to(y.dtype))


class FiLM(nn.Module):
    def __init__(self, n_norms, d, K=4, zdim=16, hidden=64, init_std=0.02):
        super().__init__()
        self.K, self.n_norms, self.d = K, n_norms, d
        self.codes = nn.Parameter(torch.randn(K, zdim))
        self.net = nn.Sequential(nn.Linear(zdim, hidden), nn.GELU(), nn.Linear(hidden, n_norms * d))
        nn.init.normal_(self.net[2].weight, std=init_std / math.sqrt(hidden))
        nn.init.zeros_(self.net[2].bias)

    def gamma(self, modes):
        """modes: LongTensor [B] -> [B, n_norms, d]"""
        return self.net(self.codes[modes]).view(len(modes), self.n_norms, self.d)


def norm_modules(model):
    """(parent, attribute name) of every RMSNorm that gets FiLM, in a fixed order."""
    out = []
    for layer in model.model.layers:
        out += [(layer, "input_layernorm"), (layer, "post_attention_layernorm")]
    out.append((model.model, "norm"))
    return out


def build(base_path, film_modes=0, lora_r=32, lora_alpha=64, device="cuda", grad_ckpt=True, lora_modes=0):
    """Frozen base + LoRA (+ FiLM if film_modes > 0; K separate adapters if lora_modes > 0).
    Returns (model, film or None, ctx)."""
    from transformers import AutoModelForCausalLM
    dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base_path, dtype=dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    ctx, film = Ctx(), None
    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name in LORA_TARGETS:
                if hasattr(parent, name):
                    lin = getattr(parent, name)
                    setattr(parent, name, MultiLoRALinear(lin, lora_r, lora_alpha, lora_modes, ctx) if lora_modes
                            else LoRALinear(lin, lora_r, lora_alpha))
    if film_modes:
        mods = norm_modules(model)
        for i, (parent, name) in enumerate(mods):
            setattr(parent, name, FiLMNorm(getattr(parent, name), i, ctx))
        film = FiLM(len(mods), model.config.hidden_size, K=film_modes)
    model.config.use_cache = False
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device)
    if film is not None:
        film.to(device)
    return model, film, ctx


def trainable(model, film):
    ps = [p for n, p in model.named_parameters() if n.endswith(".A") or n.endswith(".B")]
    for p in ps:
        p.data = p.data.float()
        p.requires_grad_(True)
    fps = list(film.parameters()) if film is not None else []
    return ps, fps


def adapter_state(model, film):
    st = {n: p.detach().cpu() for n, p in model.named_parameters() if n.endswith(".A") or n.endswith(".B")}
    if film is not None:
        st.update({f"film.{n}": p.detach().cpu() for n, p in film.named_parameters()})
    return st


def load_adapter(model, film, st):
    params = dict(model.named_parameters())
    for n, v in st.items():
        if n.startswith("film."):
            dict(film.named_parameters())[n[5:]].data.copy_(v)
        else:
            params[n].data.copy_(v)


@torch.no_grad()
def export_modes(base_path, adapter_path, out_dirs, modes, dtype=torch.bfloat16):
    """Write one plain Qwen3 checkpoint per mode (LoRA merged, FiLM folded into norm weights).

    modes: list of mode indices, or [None] for a model without FiLM / with a single adapter.
    With cfg["lora_modes"] the mode index selects which adapter is merged. Runs on CPU."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ck = torch.load(adapter_path, map_location="cpu", weights_only=False)
    st, cfg = ck["adapter"], ck["cfg"]
    model = AutoModelForCausalLM.from_pretrained(base_path, dtype=dtype)
    r, alpha = cfg["lora_r"], cfg["lora_alpha"]
    multi = cfg.get("lora_modes", 0)
    targets = []                       # (key, linear, original weight) of LoRA modules with one adapter per mode
    for li, layer in enumerate(model.model.layers):
        for pname, parent in (("self_attn", layer.self_attn), ("mlp", layer.mlp)):
            for name in LORA_TARGETS:
                key = f"model.layers.{li}.{pname}.{name}"
                if f"{key}.A" in st:
                    lin = getattr(parent, name)
                    if multi:
                        targets.append((key, lin, lin.weight.detach().clone()))
                        continue
                    delta = (alpha / r) * (st[f"{key}.B"].float() @ st[f"{key}.A"].float())
                    lin.weight.copy_((lin.weight.float() + delta).to(lin.weight.dtype))
    norms = [getattr(p, n) for p, n in norm_modules(model)]
    orig = [nm.weight.detach().clone() for nm in norms]
    film = None
    if cfg.get("film_modes"):
        film = FiLM(len(norms), model.config.hidden_size, K=cfg["film_modes"])
        film.load_state_dict({k[5:]: v for k, v in st.items() if k.startswith("film.")})
    tok = AutoTokenizer.from_pretrained(base_path)
    for out, m in zip(out_dirs, modes):
        if os.path.exists(out):
            shutil.rmtree(out)
        if multi:
            for key, lin, w0 in targets:
                delta = (alpha / r) * (st[f"{key}.B"][m].float() @ st[f"{key}.A"][m].float())
                lin.weight.copy_((w0.float() + delta).to(w0.dtype))
        elif m is not None:
            g = film.gamma(torch.tensor([m]))[0].float()          # [n_norms, d]
            for nm, w0, gl in zip(norms, orig, g):
                nm.weight.copy_((w0.float() * (1 + gl)).to(w0.dtype))
        model.save_pretrained(out, safe_serialization=True)
        tok.save_pretrained(out)
    return out_dirs
