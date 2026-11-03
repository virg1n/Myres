"""LoRA + scale-only FiLM on a Hugging Face Qwen3 model.

FiLM: after every decoder RMSNorm (input_layernorm, post_attention_layernorm) and the final norm,
    y = norm(x) * (1 + gamma_l(z)),   gamma = MLP(z),  z = one of K learned codes
(or, with continuous=True, a fresh z = N(0, I) / sqrt(zdim) for every example and every attempt, scaled by 4 inside).
Because RMSNorm is weight * normalized(x), a fixed code folds exactly into the norm weights:
    weight_l <- weight_l * (1 + gamma_l(z_k)),
so every mode exports as a plain Qwen3 checkpoint (LoRA merged too) that vLLM can serve. export_shared /
write_film_dir keep those norm weights in a shard of their own, so another code or z costs one small file.
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
    def __init__(self, n_norms, d, K=4, zdim=16, hidden=64, init_std=0.02, continuous=False):
        super().__init__()
        self.K, self.n_norms, self.d, self.zdim = K, n_norms, d, zdim
        self.codes = None if continuous else nn.Parameter(torch.randn(K, zdim))
        self.net = nn.Sequential(nn.Linear(zdim, hidden), nn.GELU(), nn.Linear(hidden, n_norms * d))
        # continuous: z = N(0, I) / sqrt(zdim) (|z| ~ 1) enters as zscale * z with the first layer of wta1 (default
        # init of a 16-input layer, bound 1/4). At zdim 16 this is the parametrization of wta1 with fresh instead of
        # learned codes; at every zdim W z starts with variance 1/3 and an Adam step moves it by the same amount.
        # The default init (bound 1 / sqrt(zdim)) would instead shrink W z by sqrt(zdim) on top of z / sqrt(zdim).
        self.zscale = 4.0 if continuous else 1.0
        if continuous:
            nn.init.uniform_(self.net[0].weight, -0.25, 0.25)
            nn.init.uniform_(self.net[0].bias, -0.25, 0.25)
        nn.init.normal_(self.net[2].weight, std=init_std / math.sqrt(hidden))
        nn.init.zeros_(self.net[2].bias)

    def sample_z(self, n, generator=None):
        """n fresh z on CPU: N(0, I) / sqrt(zdim)."""
        return torch.randn(n, self.zdim, generator=generator) / math.sqrt(self.zdim)

    def gamma_z(self, z):
        """z: [B, zdim] -> [B, n_norms, d]"""
        return self.net(z * self.zscale).view(len(z), self.n_norms, self.d)

    def gamma(self, modes):
        """modes: LongTensor [B] -> [B, n_norms, d]"""
        return self.gamma_z(self.codes[modes])


def norm_modules(model):
    """(parent, attribute name) of every RMSNorm that gets FiLM, in a fixed order."""
    out = []
    for layer in model.model.layers:
        out += [(layer, "input_layernorm"), (layer, "post_attention_layernorm")]
    out.append((model.model, "norm"))
    return out


def build(base_path, film_modes=0, lora_r=32, lora_alpha=64, device="cuda", grad_ckpt=True, lora_modes=0, film_z=0):
    """Frozen base + LoRA (+ FiLM if film_modes > 0, FiLM with continuous z of dimension film_z if film_z > 0;
    K separate adapters if lora_modes > 0). Returns (model, film or None, ctx)."""
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
    if film_modes or film_z:
        mods = norm_modules(model)
        for i, (parent, name) in enumerate(mods):
            setattr(parent, name, FiLMNorm(getattr(parent, name), i, ctx))
        film = FiLM(len(mods), model.config.hidden_size, K=film_modes, zdim=film_z or 16, continuous=bool(film_z))
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
    if cfg.get("film_z") and list(modes) != [None]:
        raise ValueError("continuous z has no modes: use export_shared + write_film_dir")
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


# ----------------------------------------------------------------------------- FiLM norms in a shard of their own
REST_SHARD, NORM_SHARD = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
INDEX = "model.safetensors.index.json"


def norm_names(n_layers):
    """State-dict names of the FiLM norms, in the order of norm_modules."""
    return [f"model.layers.{i}.{n}.weight" for i in range(n_layers)
            for n in ("input_layernorm", "post_attention_layernorm")] + ["model.norm.weight"]


def load_film(cfg, st, n_norms, d):
    """FiLM of an adapter (codes or continuous z) on CPU, or None."""
    if not (cfg.get("film_modes") or cfg.get("film_z")):
        return None
    film = FiLM(n_norms, d, K=cfg.get("film_modes") or 1, zdim=cfg.get("film_z") or 16,
                continuous=bool(cfg.get("film_z")))
    film.load_state_dict({k[5:]: v for k, v in st.items() if k.startswith("film.")})
    return film.eval()


@torch.no_grad()
def export_shared(base_path, adapter_path, out_dir, dtype=torch.bfloat16):
    """Checkpoint with LoRA merged (as export_modes) where the FiLM norms (base weights) sit in NORM_SHARD, everything
    else in REST_SHARD; written directly, so the disk never holds a second copy."""
    import json
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ck = torch.load(adapter_path, map_location="cpu", weights_only=False)
    st, cfg = ck["adapter"], ck["cfg"]
    if cfg.get("lora_modes"):
        raise ValueError("separate adapters per mode: use export_modes")
    model = AutoModelForCausalLM.from_pretrained(base_path, dtype=dtype)
    r, alpha = cfg["lora_r"], cfg["lora_alpha"]
    for li, layer in enumerate(model.model.layers):
        for pname, parent in (("self_attn", layer.self_attn), ("mlp", layer.mlp)):
            for name in LORA_TARGETS:
                key = f"model.layers.{li}.{pname}.{name}"
                if f"{key}.A" in st:
                    lin = getattr(parent, name)
                    delta = (alpha / r) * (st[f"{key}.B"].float() @ st[f"{key}.A"].float())
                    lin.weight.copy_((lin.weight.float() + delta).to(lin.weight.dtype))
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    model.config.save_pretrained(out_dir)
    if model.generation_config is not None:
        model.generation_config.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(base_path).save_pretrained(out_dir)
    tensors = {k: v.contiguous() for k, v in model.state_dict().items()}
    if getattr(model.config, "tie_word_embeddings", False):
        tensors.pop("lm_head.weight", None)            # tied to the embeddings, restored by the loader
    names = norm_names(model.config.num_hidden_layers)
    norms = {n: tensors.pop(n) for n in names}
    save_file(tensors, os.path.join(out_dir, REST_SHARD), metadata={"format": "pt"})
    save_file(norms, os.path.join(out_dir, NORM_SHARD), metadata={"format": "pt"})
    size = sum(t.numel() * t.element_size() for t in list(tensors.values()) + list(norms.values()))
    wm = {**{n: REST_SHARD for n in tensors}, **{n: NORM_SHARD for n in names}}
    with open(os.path.join(out_dir, INDEX), "w") as f:
        json.dump({"metadata": {"total_size": size}, "weight_map": wm}, f, indent=1)
    return out_dir


def _link(src, dst):
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:                     # Windows without the symlink privilege
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


@torch.no_grad()
def write_film_dir(shared, out, gamma):
    """Checkpoint of one code or z: links to the files of export_shared, the norms weight * (1 + gamma) written anew
    (as export_modes folds a code). gamma: [n_norms, d], one row of FiLM.gamma / gamma_z."""
    from safetensors.torch import load_file, save_file
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    for f in os.listdir(shared):
        if f != NORM_SHARD:
            _link(os.path.join(shared, f), os.path.join(out, f))
    w0 = load_file(os.path.join(shared, NORM_SHARD))
    names = norm_names((len(gamma) - 1) // 2)
    if set(names) != set(w0):
        raise ValueError(f"{shared}: the norms of the shared export do not match gamma {tuple(gamma.shape)}")
    save_file({n: (w0[n].float() * (1 + g.float())).to(w0[n].dtype) for n, g in zip(names, gamma)},
              os.path.join(out, NORM_SHARD), metadata={"format": "pt"})
    return out
