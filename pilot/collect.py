"""Collect natural errors of the baseline for verification training in stage 2.

Problems come from the separate "natural" split; each checkpoint solves them greedily and with
sampling. Saves natural.jsonl with verify prefixes (problem + generated solution) and labels."""
import argparse
import json
import os
import random

import torch

from . import igsm
from .common import cli_main, is_done, load_model, log, mark_done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--temps", type=float, nargs="+", default=[0.0, 0.8])
    ap.add_argument("--gen_bs", type=int, default=512)
    ap.add_argument("--seed", type=int, default=77)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if is_done(a.out):
        return
    model, meta = load_model(a.ckpts[0])
    op_lo, op_max = max(meta["op_lo"], meta["op_max"] // 2), meta["op_max"]
    probs = igsm.fixed_set(a.seed, "natural", a.n, op_lo, op_max, model.cfg.max_len)
    prompts = [igsm.solve_prompt(p) for p in probs]
    wrong, right = [], []
    stats = {}
    for ci, path in enumerate(a.ckpts):
        model, _ = load_model(path)
        model.eval()
        for t in a.temps:
            outs = model.generate(prompts, temperature=t, eos_id=igsm.EOS, batch_size=a.gen_bs,
                                  seed=a.seed + ci * 10 + int(t * 100))
            n_err = 0
            for p, o in zip(probs, outs):
                chk = igsm.check_solution(p, o)
                pre = igsm.verify_prefix(p, o)
                if len(pre) + 4 > model.cfg.max_len:
                    continue
                rec = {"prefix": pre, "err": chk["first_err"] if chk["has_error"] else None, "op": p.op}
                (wrong if chk["has_error"] else right).append(rec)
                n_err += chk["has_error"]
            stats[f"{os.path.basename(path)}@T{t}"] = n_err / len(probs)
            log(f"{path} T={t}: error rate {n_err / len(probs):.3f}")
    rng = random.Random(a.seed)
    rng.shuffle(right)
    right = right[:max(len(wrong), 1000)]
    with open(os.path.join(a.out, "natural.jsonl"), "w") as f:
        for r in wrong + right:
            f.write(json.dumps(r) + "\n")
    mark_done(a.out, {"n_wrong": len(wrong), "n_right": len(right), "error_rate": stats})
    log(f"natural errors: {len(wrong)} wrong, {len(right)} correct")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    cli_main(main)
