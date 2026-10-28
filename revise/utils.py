import copy
import json
import math
import os
import shutil
import time

import torch
import torch.distributed as dist
import yaml

DEFAULTS = {
    "name": "run",
    "out_dir": "runs/run",
    "seed": 1234,
    "disk_limit_gb": 35.0,
    "data": {
        "gen": {},
        "op_train": [2, 15],
        "op_rollout": [6, 15],
        "op_test_id": [10, 15],
        "op_test_ood": [16, 20],
        "strict_necessary": True,
    },
    "model": {"d_model": 384, "n_layers": 6, "n_heads": 6, "d_ff": 1024, "dep_dim": 64, "max_pos": 2048},
    "revision": {
        "window": 64, "slack": 4, "slack_min_len": 3, "identity_w": 0.2,
        "mask_patterns": [0.5, 0.2, 0.15, 0.15],  # dag, span, line, random
        "dep_target": "confdrop",  # confdrop | oracle
        "confdrop_delta": 2.0, "confdrop_iters": 4,
        "neg_window_prob": 0.5, "lm_on_self": False,
    },
    "train": {
        "batch_size": 64, "plain_steps": 20000, "branch_frac": 0.5,
        "lr": 1.0e-3, "min_lr": 1.0e-4, "warmup_frac": 0.02, "rewarmup_steps": 200,
        "weight_decay": 0.1, "betas": [0.9, 0.95], "grad_clip": 1.0,
        "windows_per_seq": 2, "masked_window_prob": 0.5, "self_error_frac": 0.4,
        "loss_w": {"lm": 1.0, "err": 1.0, "fix": 1.0, "dep": 1.0},
        "err_pos_weight": 1.0, "dep_warmup_steps": 300,
        "rollout_every": 50, "rollout_problems": 256, "rollout_batch": 256, "rollout_temps": [0.0, 0.7, 1.0],
        "rollout_max_new": 600, "buffer_size": 30000, "min_buffer": 512,
        "num_workers": 6, "log_every": 50, "save_every": 1000, "eval_every": 2000, "eval_n": 256, "eval_revision_n": 64,
        "amp": "bf16", "compile": False, "max_hours": 1000,
    },
    "eval": {
        "n_test": 1000, "batch_size": 128, "plain_batch_size": 256,
        "variants": ["plain", "fix_only", "erase_suffix", "dep_head", "oracle_dep", "dep_head_lines",
                     "random_erase", "confdrop_measure", "revision_model_no_revision"],
        "policy": {"tau_err": 0.5, "tau_dep": 0.5, "max_fixes": 8, "check_every": 1, "max_sol_len": 600,
                   "diagnostics": True},
        "sweep_max_fixes": [],
        "majority_k": "auto", "majority_k_max": 32, "majority_temp": 0.7,
    },
}


def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


def load_config(path, overrides=()):
    cfg = copy.deepcopy(DEFAULTS)
    with open(path) as f:
        deep_update(cfg, yaml.safe_load(f) or {})
    for o in overrides:
        key, val = o.split("=", 1)
        d = cfg
        ks = key.split(".")
        for k in ks[:-1]:
            d = d.setdefault(k, {})
        d[ks[-1]] = yaml.safe_load(val)
    if cfg.get("out_dir") in (None, "", DEFAULTS["out_dir"]):
        cfg["out_dir"] = os.path.join("runs", cfg["name"])
    return cfg


# ---------------------------------------------------------------------------
# distributed
# ---------------------------------------------------------------------------

def init_dist():
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend)
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world, local = 0, 1, 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")
    return rank, world, device


def is_dist():
    return dist.is_available() and dist.is_initialized()


def all_reduce_sum(x, device):
    t = torch.tensor(float(x), device=device, dtype=torch.float64)
    if is_dist():
        dist.all_reduce(t)
    return t.item()


def barrier():
    if is_dist():
        dist.barrier()


def cleanup_dist():
    if is_dist():
        dist.destroy_process_group()


def amp_dtype(cfg):
    a = cfg["train"].get("amp", "bf16")
    return {"bf16": torch.bfloat16, "fp16": torch.float16}.get(a, None)


# ---------------------------------------------------------------------------
# logging / disk / checkpoints
# ---------------------------------------------------------------------------

class Logger:
    def __init__(self, path, rank):
        self.rank = rank
        self.path = path
        self.t0 = time.time()
        if rank == 0:
            os.makedirs(os.path.dirname(path), exist_ok=True)

    def log(self, d, echo=True):
        if self.rank != 0:
            return
        d = dict(d)
        d["time_min"] = round((time.time() - self.t0) / 60, 2)
        with open(self.path, "a") as f:
            f.write(json.dumps(d) + "\n")
        if echo:
            print(" | ".join(f"{k}={_fmt(v)}" for k, v in d.items()), flush=True)


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def dir_size_gb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 1e9


def check_disk(root, limit_gb, extra_gb=0.0):
    """Abort before writing if the run directory would exceed the storage limit."""
    used = dir_size_gb(root) if os.path.exists(root) else 0.0
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(root)) or ".").free / 1e9
    if used + extra_gb > limit_gb:
        raise RuntimeError(f"disk guard: {root} uses {used:.2f} GB (+{extra_gb:.2f} GB planned) "
                           f"> limit {limit_gb} GB. Delete old runs or raise disk_limit_gb.")
    if free < extra_gb + 1.0:
        raise RuntimeError(f"disk guard: only {free:.2f} GB free on device")
    return used


def save_checkpoint(path, obj, cfg):
    est = sum(t.numel() * t.element_size() for t in _tensors(obj)) / 1e9
    check_disk(cfg["out_dir"], cfg["disk_limit_gb"], extra_gb=est)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _tensors(o):
    if torch.is_tensor(o):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from _tensors(v)
    elif isinstance(o, (list, tuple)):
        for v in o:
            yield from _tensors(v)


def lr_at(frac, cfg):
    t = cfg["train"]
    if frac < t["warmup_frac"]:
        return t["lr"] * frac / max(1e-9, t["warmup_frac"])
    p = min(1.0, (frac - t["warmup_frac"]) / max(1e-9, 1 - t["warmup_frac"]))
    return t["min_lr"] + 0.5 * (t["lr"] - t["min_lr"]) * (1 + math.cos(math.pi * p))


def unwrap(model):
    return model.module if hasattr(model, "module") else model
