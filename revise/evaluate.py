"""Evaluate all variants on an in-distribution and an out-of-distribution
(harder, more operations) iGSM test set. Problems are sharded over ranks.

  torchrun --nproc_per_node=4 -m revise.evaluate --config configs/small.yaml
"""

import argparse
import json
import math
import os
import time

import torch
import torch.distributed as dist

from .decode import VARIANTS, decode_majority, decode_plain, decode_revision, make_policy
from .igsm.generator import GenConfig, make_test_set
from .igsm.tokenizer import Tokenizer
from .model import ModelConfig, RevisionLM
from .utils import amp_dtype, barrier, cleanup_dist, init_dist, is_dist, load_config


def load_model(path, cfg, tok, device):
    ck = torch.load(path, map_location="cpu")
    m = RevisionLM(ModelConfig.from_dict(cfg["model"], len(tok)))
    m.load_state_dict(ck["model"])
    return m.to(device).eval()


def gather(results, world):
    if world == 1:
        return results
    out = [None] * world
    dist.all_gather_object(out, results)
    return [r for part in out for r in part]


def summarize(res, plain_flops=None):
    n = len(res)
    acc = sum(r["correct"] for r in res) / n
    s = {
        "n": n,
        "acc": acc,
        "acc_se": math.sqrt(acc * (1 - acc) / n),
        "sol_correct": sum(r["sol_correct"] for r in res) / n,
        "complete": sum(r["complete"] for r in res) / n,
        "steps": sum(r["steps"] for r in res) / n,
        "tokens": sum(r["tokens"] for r in res) / n,
        "flops": sum(r.get("flops", 0.0) for r in res) / n,
        "fixes": sum(r["fixes"] for r in res) / n,
        "fills": sum(r["fills"] for r in res) / n,
        "erased_per_fix": sum(r["erased"] for r in res) / max(1, sum(r["fixes"] for r in res)),
    }
    if plain_flops:
        s["flops_x_plain"] = s["flops"] / plain_flops
    ev = [e for r in res for e in r["events"] if "true_err" in e]
    if ev:
        s["fix_precision"] = sum(e["true_err"] for e in ev) / len(ev)
        s["fix_at_first_err"] = sum(e["is_first_err"] for e in ev) / len(ev)
        s["fix_token_valid"] = sum(e["fix_valid"] for e in ev) / max(1, sum(e["true_err"] for e in ev))
        tp, ne, nt = sum(e["tp"] for e in ev), sum(e["n_erase"] for e in ev), sum(e["n_true_dep"] for e in ev)
        s["erase_precision_vs_true_deps"] = tp / max(1, ne)
        s["erase_recall_vs_true_deps"] = tp / max(1, nt)
    by_op = {}
    for r in res:
        by_op.setdefault(r["op"], []).append(r["correct"])
    s["acc_by_op"] = {str(k): sum(v) / len(v) for k, v in sorted(by_op.items())}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    rank, world, device = init_dist()
    amp = amp_dtype(cfg) if device.type == "cuda" else None
    tok = Tokenizer()
    e = cfg["eval"]
    out_dir = os.path.join(cfg["out_dir"], "results")
    os.makedirs(out_dir, exist_ok=True)

    plain = load_model(os.path.join(cfg["out_dir"], "plain", "final.pt"), cfg, tok, device)
    rev_path = os.path.join(cfg["out_dir"], "revise", "final.pt")
    rev = load_model(rev_path, cfg, tok, device) if os.path.exists(rev_path) else None
    N = plain.n_params()
    gcfg = GenConfig.from_dict(cfg["data"]["gen"])
    variants = args.variants or e["variants"]
    max_len = e["policy"]["max_sol_len"]

    test_sets = {"id": (cfg["data"]["op_test_id"], cfg["seed"] + 1000003),
                 "ood": (cfg["data"]["op_test_ood"], cfg["seed"] + 2000003)}
    summary = {}
    for set_name, (op, seed) in test_sets.items():
        probs = make_test_set(gcfg, op, e["n_test"], seed, tok)
        shard = list(range(rank, len(probs), world))
        mine = [probs[i] for i in shard]
        summary[set_name] = {}
        os.makedirs(os.path.join(out_dir, set_name), exist_ok=True)

        def run(tag, fn):
            path = os.path.join(out_dir, set_name, f"{tag}.json")
            if os.path.exists(path):
                with open(path) as f:
                    return json.load(f)["results"]
            t0 = time.time()
            res = fn()
            for r in res:
                r["idx"] = shard[r["idx"]]
            res = sorted(gather(res, world), key=lambda r: r["idx"])
            if rank == 0:
                with open(path, "w") as f:
                    json.dump({"tag": tag, "results": res}, f)
                acc = sum(r["correct"] for r in res) / len(res)
                print(f"[{set_name}] {tag:34s} acc={acc:.4f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
            barrier()
            return res

        plain_res = run("plain", lambda: decode_plain(plain, mine, tok, device, e["plain_batch_size"], amp,
                                                         max_len, n_params=N))
        plain_flops = sum(r["flops"] for r in plain_res) / len(plain_res)
        summary[set_name]["plain"] = summarize(plain_res, plain_flops)

        for v in variants:
            if v == "plain":
                continue
            if rev is None:
                if rank == 0:
                    print(f"no revision model found at {rev_path}; skipping {v}")
                continue
            pol = make_policy(v, {**e["policy"], "window": cfg["revision"]["window"],
                                  "slack": cfg["revision"]["slack"],
                                  "slack_min_len": cfg["revision"]["slack_min_len"],
                                  "confdrop_delta": cfg["revision"]["confdrop_delta"],
                                  "confdrop_iters": cfg["revision"]["confdrop_iters"]})
            res = run(v, lambda: decode_revision(rev, mine, pol, tok, device, e["batch_size"], amp,
                                                 seed=cfg["seed"], n_params=N))
            summary[set_name][v] = summarize(res, plain_flops)

            for mf in e.get("sweep_max_fixes") or []:
                if v in ("revision_model_no_revision",):
                    continue
                pol_s = make_policy(v, {**vars(pol), "max_fixes": mf})
                res = run(f"{v}@maxfix{mf}", lambda: decode_revision(rev, mine, pol_s, tok, device, e["batch_size"],
                                                                     amp, seed=cfg["seed"], n_params=N))
                summary[set_name][f"{v}@maxfix{mf}"] = summarize(res, plain_flops)

        # compute-matched baseline: plain model + majority vote
        k = e.get("majority_k", "auto")
        if k == "auto":
            ref = summary[set_name].get("dep_head")
            k = int(math.ceil(ref["flops_x_plain"])) if ref else 0
            k = max(1, min(k, e.get("majority_k_max", 32)))
        if k and int(k) > 1:
            k = int(k)
            res = run(f"plain_majority@{k}", lambda: decode_majority(plain, mine, tok, device, k,
                                                                    e["plain_batch_size"], amp, max_len,
                                                                    e.get("majority_temp", 0.7), N))
            summary[set_name][f"plain_majority@{k}"] = summarize(res, plain_flops)

    if rank == 0:
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        from .report import write_report
        write_report(cfg)
    barrier()
    cleanup_dist()


if __name__ == "__main__":
    main()
