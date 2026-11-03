"""Evaluation at large k on test (phase1/zrun.py): every attempt of a model with continuous z gets its own z.

  z       attempt j of every problem uses z_j (z_0..z_{k-1} drawn once from --seed, saved in z.pt): one checkpoint
          per z, 1 sample each, greedy (T = 0) for the first --greedy z
  modes   FiLM codes or separate adapters (LoRA-MCL): k / M samples per mode, greedy for every mode
  single  no modes: k samples, one greedy
Attempt j (mode m) samples with the vLLM seed --seed + 1000 * (j + 1). A checkpoint of a z or a FiLM code is the
shared export (LoRA merged) with only the 57 norm weights written anew (film.write_film_dir): seconds and almost no
disk per z; separate adapters are exported one at a time (3.5 GB). Outputs per unit t: solve_<t>_test.jsonl.gz
(n samples), greedy_<t>_test.jsonl.gz; then eval.json and DONE. keval.json pins the adapter, the data and the sampling
settings: a restart with others is refused instead of mixing their outputs."""
import argparse
import json
import os
import shutil
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase1 import film as FM  # noqa: E402
from phase1.evaluate import CKPT_GB, gen, log, wait_disk  # noqa: E402


def units(cfg, k, greedy):
    """(kind, [(tag, samples, greedy?)])."""
    if cfg.get("prefix_modes"):
        raise SystemExit("ModC prefixes are not supported by keval")
    if cfg.get("film_z"):
        return "z", [(j, 1, j < greedy) for j in range(k)]
    M = cfg.get("film_modes", 0) or cfg.get("lora_modes", 0)
    if M:
        if k % M:
            raise SystemExit(f"k = {k} is not a multiple of the {M} modes")
        return "modes", [(m, k // M, True) for m in range(M)]
    return "single", [(0, k, True)]


def unit_done(out, t, gr):
    return os.path.exists(os.path.join(out, f"solve_{t}_test.jsonl.gz")) and \
        (not gr or os.path.exists(os.path.join(out, f"greedy_{t}_test.jsonl.gz")))


def check_manifest(out, want):
    """Outputs of a restarted evaluation are reused only with the same adapter, data and sampling settings."""
    f = os.path.join(out, "keval.json")
    if os.path.exists(f):
        have = json.load(open(f))
        diff = {k: (have.get(k), v) for k, v in want.items() if have.get(k) != v}
        if diff:
            raise SystemExit(f"{out} was started with other settings (was, now): {diff}; use another --out")
        return
    with open(f + ".tmp", "w") as g:
        json.dump(want, g, indent=1)
    os.replace(f + ".tmp", f)


def the_z(out, film, k, seed):
    """z_0..z_{k-1}: drawn once, the same after a restart (and refused if the file disagrees)."""
    f = os.path.join(out, "z.pt")
    if os.path.exists(f):
        Z = torch.load(f)
        if tuple(Z.shape) != (k, film.zdim):
            raise SystemExit(f"{f} has z of shape {tuple(Z.shape)}, asked ({k}, {film.zdim})")
        return Z
    Z = film.sample_z(k, torch.Generator().manual_seed(seed))
    torch.save(Z, f)
    return Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", default="")
    ap.add_argument("--k", type=int, default=32, help="attempts per problem")
    ap.add_argument("--greedy", type=int, default=8, help="continuous z: greedy solutions for the first z")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--max_tokens", type=int, default=1200)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    check_manifest(a.out, {"adapter": os.path.abspath(a.adapter), "adapter_bytes": os.path.getsize(a.adapter),
                           "adapter_mtime": int(os.path.getmtime(a.adapter)), "base": os.path.abspath(a.base),
                           "test": os.path.abspath(os.path.join(a.data, "test.jsonl")),
                           "n_test": sum(1 for _ in open(os.path.join(a.data, "test.jsonl"))), "k": a.k,
                           "greedy": a.greedy, "seed": a.seed, "max_tokens": a.max_tokens})
    if os.path.exists(os.path.join(a.out, "DONE")):
        return
    ck = torch.load(a.adapter, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    kind, U = units(cfg, a.k, a.greedy)
    tmp = a.tmp or os.path.join(a.out, "tmp_export")
    shutil.rmtree(tmp, ignore_errors=True)         # left behind if a previous attempt was killed
    os.makedirs(tmp, exist_ok=True)
    todo = [u for u in U if not unit_done(a.out, u[0], u[2])]
    try:
        if todo:
            from transformers import AutoConfig
            mc = AutoConfig.from_pretrained(a.base)
            film = FM.load_film(cfg, ck["adapter"], 2 * mc.num_hidden_layers + 1, mc.hidden_size)
            Z = the_z(a.out, film, a.k, a.seed) if kind == "z" else None
            shared = None
            if film is not None:
                wait_disk(tmp, CKPT_GB + 2)
                shared = FM.export_shared(a.base, a.adapter, os.path.join(tmp, "shared"))
                log("shared export written")
            d = os.path.join(tmp, "model")       # one path for every unit: vLLM keys its compile cache by the path
            for t, n, gr in todo:
                t0 = time.time()
                if film is not None:
                    with torch.no_grad():
                        g = film.gamma_z(Z[t:t + 1])[0] if kind == "z" else film.gamma(torch.tensor([t]))[0]
                    FM.write_film_dir(shared, d, g)
                else:
                    wait_disk(tmp, CKPT_GB + 2)
                    FM.export_modes(a.base, a.adapter, [d], [t if kind == "modes" else None])
                gen("k_solve", d, a.data, a.out, ["--tag", str(t), "--n", str(n), "--greedy", str(int(gr)),
                                                  "--seed", str(a.seed + 1000 * (t + 1))], max_tokens=a.max_tokens)
                shutil.rmtree(d, ignore_errors=True)
                log(f"unit {t + 1}/{len(U)} ({kind}, {n} sample(s){', greedy' if gr else ''}) in "
                    f"{(time.time() - t0) / 60:.1f} min")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    with open(os.path.join(a.out, "eval.json"), "w") as f:
        json.dump({"kind": kind, "k": a.k, "units": len(U), "n": U[0][1], "greedy": [t for t, _, gr in U if gr],
                   "zdim": cfg.get("film_z", 0), "seed": a.seed, "max_tokens": a.max_tokens}, f)
    open(os.path.join(a.out, "DONE"), "w").close()
    log("k-eval done", a.out)


if __name__ == "__main__":
    main()
