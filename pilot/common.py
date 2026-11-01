import json
import os
import random
import shutil
import sys
import time

import numpy as np
import torch

from . import igsm
from .model import GPT, ModelCfg, autocast  # noqa: F401  (re-exported)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s):
    random.seed(s)
    np.random.seed(s % (2 ** 32))
    torch.manual_seed(s)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def model_cfg(size):
    if size == "tiny":     # for smoke tests only
        return ModelCfg(vocab=igsm.VOCAB_SIZE, n_layer=2, n_head=2, d=64)
    if size == "small":    # CPU sanity runs
        return ModelCfg(vocab=igsm.VOCAB_SIZE, n_layer=4, n_head=4, d=128)
    if size == "20m":
        return ModelCfg(vocab=igsm.VOCAB_SIZE)
    raise ValueError(size)


def save_model(path, model, meta):
    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    tmp = path + ".tmp"
    torch.save({"model": m.state_dict(), "cfg": m.cfg.to_dict(), "meta": meta}, tmp)
    os.replace(tmp, path)


def load_model(path, dev=None):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = GPT(ModelCfg(**ck["cfg"]))
    m.load_state_dict(ck["model"])
    m.to(dev or device())
    return m, ck["meta"]


def mode_z(model, z_mode, mode, n, gen=None):
    """z for `n` rows under mode index `mode` (fix: code; cont: fresh N(0,I); none: None)."""
    dev = model.emb.weight.device
    if z_mode == "none":
        return None
    if z_mode == "fix":
        return model.code_z(torch.full((n,), mode, dtype=torch.long, device=dev)).detach()
    return torch.randn(n, model.cfg.zdim, device=dev, generator=gen)


def disk_usage_gb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 1e9


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def mark_done(out, result=None):
    if result is not None:
        write_json(os.path.join(out, "result.json"), result)
    open(os.path.join(out, "DONE"), "w").close()


def is_done(out):
    return os.path.exists(os.path.join(out, "DONE"))


def cli_main(fn):
    try:
        fn()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
