"""Stage 2: WTA by correctness of self-generated solutions.

For each problem every mode (fix: code k; cont: fresh z; none: plain sample) generates a solution.
W = modes whose final answer is correct and whose steps all check out.
  --weights unique : each winner gets weight 1/|W|   (a mode that alone solves a problem gets 1)
  --weights uniform: each winner gets weight 1/K     (= rejection-sampling fine-tuning per mode)
W empty -> teacher-forced reference solution with relaxed WTA (weight 1).
Verification examples: natural baseline errors + injected errors, trained with relaxed WTA.
"""
import argparse
import csv
import json
import math
import os
import random
import time

import torch

from . import igsm
from .common import (autocast, cli_main, device, is_done, load_model, log, mark_done, mode_z,
                     save_model, set_seed)
from .data import collate
from .model import per_example_nll
from .train import choose_z, make_optimizer


def micro_batches(examples, bs):
    examples = sorted(examples, key=lambda e: len(e[0]))
    return [examples[i:i + bs] for i in range(0, len(examples), bs)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--natural", required=True, help="natural.jsonl from collect")
    ap.add_argument("--weights", choices=["unique", "uniform"], default="unique")
    ap.add_argument("--problems", type=int, default=80000)
    ap.add_argument("--P", type=int, default=128, help="problems per step")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--verify_frac", type=float, default=0.3)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--balance", type=float, default=1.25)
    ap.add_argument("--micro_bs", type=int, default=96)
    ap.add_argument("--gen_bs", type=int, default=512)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--ckpt_minutes", type=float, default=20)
    ap.add_argument("--log_every", type=int, default=20)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if is_done(a.out):
        return
    dev = device()
    set_seed(a.seed)
    model, meta = load_model(a.init, dev)
    zm, K = meta["z"], meta.get("K", 4)
    opt = make_optimizer(model, a.lr, a.wd)
    natural = [json.loads(l) for l in open(a.natural)]
    nat_wrong = [r for r in natural if r["err"] is not None]
    nat_right = [r for r in natural if r["err"] is None]
    steps = max(1, a.problems // a.P)
    st = {"step": 0}
    last = os.path.join(a.out, "last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        st = ck["state"]
        log(f"resumed stage2 at step {st['step']}")
    rng = random.Random(a.seed * 7919 + st["step"])
    gen = torch.Generator(device=dev)
    gen.manual_seed(a.seed + st["step"])
    logf = open(os.path.join(a.out, "log.csv"), "a", newline="")
    w = csv.writer(logf)
    if st["step"] == 0:
        w.writerow(["step", "lr", "solve_rate", "mean_winners", "all_fail", "unique_share", "loss"] +
                   [f"acc_mode{k}" for k in range(K)])
    n_ver = round(a.P * a.verify_frac / (1 - a.verify_frac))
    agg = {"solve": 0.0, "nw": 0.0, "fail": 0.0, "uniq": 0.0, "loss": 0.0, "n": 0, "acc": [0.0] * K}
    t_ckpt = time.time()
    log(f"stage2 {a.out}: z={zm} weights={a.weights} steps={steps}")
    while st["step"] < steps:
        frac = st["step"] / steps
        lr = a.lr * min(1.0, (st["step"] + 1) / 20) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups:
            g["lr"] = lr
        probs = [igsm.sample_problem(rng, "train", meta["op_lo"], meta["op_max"], model.cfg.max_len)
                 for _ in range(a.P)]
        prompts = [igsm.solve_prompt(p) for p in probs]
        # ---- generate K solutions per problem
        model.eval()
        rows, zs, codes = [], [], []
        for k in range(K):
            rows += prompts
            codes += [k] * a.P
            z = mode_z(model, zm, k, a.P, gen)
            if z is not None:
                zs.append(z)
        Z = torch.cat(zs) if zs else None
        outs = model.generate(rows, z=Z, temperature=a.temp, eos_id=igsm.EOS, batch_size=a.gen_bs,
                              seed=a.seed * 1000003 + st["step"])
        model.train()
        own, wta = [], []          # own: (ids, loss_from, False, weight, row) ; wta: (ids, lf, verify, w)
        for i, p in enumerate(probs):
            winners = []
            for k in range(K):
                r = k * a.P + i
                chk = igsm.check_solution(p, outs[r])
                if not chk["has_error"]:
                    winners.append(r)
                agg["acc"][k] += chk["correct"] / a.P
            agg["nw"] += len(winners)
            agg["solve"] += bool(winners)
            agg["uniq"] += len(winners) == 1
            if winners:
                wt = 1.0 / len(winners) if a.weights == "unique" else 1.0 / K
                for r in winners:
                    ids = prompts[i] + outs[r] + [igsm.EOS]
                    if len(ids) <= model.cfg.max_len:
                        own.append((ids, len(prompts[i]), False, wt, r))
            else:
                agg["fail"] += 1
                ids, lf = igsm.solve_sequence(p, rng)
                wta.append((ids, lf, False, 1.0))
        for v in range(n_ver):
            if v % 2 == 0 and nat_wrong:
                rec = rng.choice(nat_wrong if rng.random() < 0.5 else (nat_right or nat_wrong))
                ids = rec["prefix"] + igsm.verdict_tokens(rec["err"])
                wta.append((ids, len(rec["prefix"]) + 1, True, 1.0))
            else:
                p = igsm.sample_problem(rng, "train", meta["op_lo"], meta["op_max"], model.cfg.max_len)
                ids, lf = igsm.verify_sequence(p, rng, with_error=rng.random() < 0.5)
                wta.append((ids, lf, True, 1.0))
        # ---- train
        denom = a.P + n_ver
        total = 0.0
        for mb in micro_batches(own, a.micro_bs):
            b = collate(mb)
            x, y, wv = b["x"].to(dev), b["y"].to(dev), b["w"].to(dev)
            rws = torch.tensor([e[4] for e in mb], device=dev)
            if zm == "fix":
                z = model.code_z(rws // a.P)
            elif zm == "cont":
                z = Z[rws]
            else:
                z = None
            with autocast(dev):
                nll = per_example_nll(model(x, z), y)
            loss = (nll * wv).sum() / denom
            loss.backward()
            total += float(loss.detach())
        for mb in micro_batches(wta, a.micro_bs):
            b = collate(mb)
            x, y, wv = b["x"].to(dev), b["y"].to(dev), b["w"].to(dev)
            with autocast(dev):
                z, _ = choose_z(model, model, x, y, zm, "wta", K, a.eps, a.balance)
                nll = per_example_nll(model(x, z), y)
            loss = (nll * wv).sum() / denom
            loss.backward()
            total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        st["step"] += 1
        agg["loss"] += total
        agg["n"] += 1
        if st["step"] % a.log_every == 0 or st["step"] == steps:
            n = agg["n"] * a.P
            row = [st["step"], f"{lr:.2e}", agg["solve"] / n, agg["nw"] / n, agg["fail"] / n,
                   agg["uniq"] / n, agg["loss"] / agg["n"]] + [x / agg["n"] for x in agg["acc"]]
            w.writerow([f"{x:.4f}" if isinstance(x, float) else x for x in row])
            logf.flush()
            log(f"stage2 step {st['step']}/{steps} any-mode solved {agg['solve'] / n:.3f} "
                f"winners/problem {agg['nw'] / n:.2f} unique {agg['uniq'] / n:.3f} "
                f"acc per mode {[round(x / agg['n'], 3) for x in agg['acc']]}")
            agg = {"solve": 0.0, "nw": 0.0, "fail": 0.0, "uniq": 0.0, "loss": 0.0, "n": 0, "acc": [0.0] * K}
        if time.time() - t_ckpt > a.ckpt_minutes * 60:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": st}, last + ".tmp")
            os.replace(last + ".tmp", last)
            t_ckpt = time.time()
    save_model(os.path.join(a.out, "final.pt"), model,
               dict(meta, stage2=a.weights, init=a.init, stage2_problems=steps * a.P))
    if os.path.exists(last):
        os.remove(last)
    mark_done(a.out, {"steps": steps})
    log("done", a.out)


if __name__ == "__main__":
    cli_main(main)
