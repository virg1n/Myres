"""Evaluate a phase-1 adapter (and stage-2 helpers): export modes -> vLLM jobs of phase0.gen.

Model with K modes: every mode solves val and test (2 samples, neutral prompt), then every mode
verifies sample 0 of every mode on test (Yes/No).  Model without modes: 8 samples + one verifier.
ModC: the same as K modes, from one checkpoint; a mode is the "[Mode k]" prefix of the prompt.
Exported checkpoints (~3.5 GB per mode) live in a temporary directory and are deleted at the end."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase1 import film as FM  # noqa: E402

PY = sys.executable
CKPT_GB = 3.5


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def gen(job, model, data, out, extra=(), eager=0, max_tokens=1200):
    cmd = [PY, "-m", "phase0.gen", job, "--model", model, "--data", data, "--out", out, "--eager", str(eager),
           "--max_tokens", str(max_tokens), *extra]
    env = dict(os.environ)
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    log("run", " ".join(cmd[3:5]), " ".join(extra))
    rc = subprocess.call(cmd, env=env)
    if rc != 0:
        raise SystemExit(f"phase0.gen {job} failed (exit {rc})")


def wait_disk(path, need_gb, timeout_h=1.0):
    """Other evaluations delete their checkpoints when they finish; wait for them rather than fill the disk."""
    t0 = time.time()
    while shutil.disk_usage(path).free / 1e9 < need_gb:
        if time.time() - t0 > timeout_h * 3600:
            raise SystemExit(f"less than {need_gb:.1f} GB free for {timeout_h} h")
        log(f"waiting for {need_gb:.1f} GB of free disk")
        time.sleep(60)


def export(base, adapter, tmp, modes):
    dirs = [os.path.join(tmp, f"mode{m if m is not None else 'X'}") for m in modes]
    wait_disk(tmp, CKPT_GB * len(dirs) + 2)
    t0 = time.time()
    FM.export_modes(base, adapter, dirs, modes)
    log(f"exported {len(dirs)} checkpoint(s) in {(time.time() - t0) / 60:.1f} min")
    return dirs


def adapter_cfg(adapter):
    return torch.load(adapter, map_location="cpu", weights_only=False)["cfg"]


def n_modes(adapter):
    """Modes that need their own checkpoint (FiLM codes or separate adapters)."""
    cfg = adapter_cfg(adapter)
    return cfg.get("film_modes", 0) or cfg.get("lora_modes", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", default="")
    ap.add_argument("--modes", type=int, default=4, help="how many modes to evaluate")
    ap.add_argument("--sets", default="val,test")
    ap.add_argument("--seed", type=int, default=17, help="mode k samples with seed + 1000*(k+1)")
    ap.add_argument("--max_tokens", type=int, default=1200)
    # one at a time by default: on a volume with a quota the free space reported by the OS is not the real limit
    ap.add_argument("--min_free_gb_all", type=float, default=1e12,
                    help="export all modes at once only if this much disk is free; else one at a time")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if os.path.exists(os.path.join(a.out, "DONE")):
        return
    tmp = a.tmp or os.path.join(a.out, "tmp_export")
    shutil.rmtree(tmp, ignore_errors=True)         # left behind if a previous attempt was killed
    os.makedirs(tmp, exist_ok=True)
    cfg = adapter_cfg(a.adapter)
    prefix_modes = cfg.get("prefix_modes", 0)
    K = n_modes(a.adapter) or prefix_modes
    try:
        if prefix_modes:
            M = min(K, a.modes)
            solve_todo = [k for k in range(M) if not all(os.path.exists(os.path.join(a.out, f"solve_{k}_{s}.jsonl.gz"))
                                                         for s in a.sets.split(","))]
            verify_todo = [j for j in range(M) if "test" in a.sets.split(",") and
                           not os.path.exists(os.path.join(a.out, f"verify_{j}.json"))]
            if solve_todo or verify_todo:
                d = export(a.base, a.adapter, tmp, [None])[0]
                for k in solve_todo:
                    gen("mode_solve", d, a.data, a.out, ["--tag", str(k), "--sets", a.sets, "--n", "2", "--prefix_mode", str(k),
                                                         "--seed", str(a.seed + 1000 * (k + 1))],
                        max_tokens=a.max_tokens)
                for j in verify_todo:
                    gen("mode_verify", d, a.data, a.out, ["--tag", str(j), "--members", str(M), "--solve_dir", a.out,
                                                          "--prefix_mode", str(j)], eager=1)
        elif K == 0:
            if not os.path.exists(os.path.join(a.out, "verify_single.json")):
                d = export(a.base, a.adapter, tmp, [None])[0]
                gen("single_eval", d, a.data, a.out, ["--sets", a.sets, "--seed", str(a.seed + 1000)],
                    max_tokens=a.max_tokens)
        else:
            M = min(K, a.modes)
            all_at_once = shutil.disk_usage(tmp).free / 1e9 > a.min_free_gb_all
            dirs = export(a.base, a.adapter, tmp, list(range(M))) if all_at_once else None
            for k in range(M):
                if all(os.path.exists(os.path.join(a.out, f"solve_{k}_{s}.jsonl.gz")) for s in a.sets.split(",")):
                    continue
                d = dirs[k] if dirs else export(a.base, a.adapter, tmp, [k])[0]
                gen("mode_solve", d, a.data, a.out, ["--tag", str(k), "--sets", a.sets, "--n", "2",
                                                     "--seed", str(a.seed + 1000 * (k + 1))],
                    max_tokens=a.max_tokens)
                if not dirs:
                    shutil.rmtree(d, ignore_errors=True)
            if "test" in a.sets.split(","):
                for j in range(M):
                    if os.path.exists(os.path.join(a.out, f"verify_{j}.json")):
                        continue
                    d = dirs[j] if dirs else export(a.base, a.adapter, tmp, [j])[0]
                    gen("mode_verify", d, a.data, a.out, ["--tag", str(j), "--members", str(M),
                                                          "--solve_dir", a.out], eager=1)
                    if not dirs:
                        shutil.rmtree(d, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    with open(os.path.join(a.out, "eval.json"), "w") as f:
        json.dump({"kind": "single" if K == 0 else "modes", "modes": 4 if K == 0 else min(K, a.modes),
                   "sets": a.sets.split(","), "verify": "test" in a.sets.split(",")}, f)
    open(os.path.join(a.out, "DONE"), "w").close()
    log("eval done", a.out)


if __name__ == "__main__":
    main()
