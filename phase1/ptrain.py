"""Training of shared + private LoRA (phase1/priv.py) for the private-share sweep (phase1/prun.py).

  --task math  the phase-1 data and recipe of wta1: one reference per problem + Yes/No verifier examples (ver_ratio),
               P problems per step, cosine LR with 30 warm-up steps, FiLM LR 5x; examples built by phase1.train
  --task mt    sentence pairs (rows {src, tgt}) with the prompt of LoRA-MCL / ALMA,
               "Translate this from English to German:\\nEnglish: <src>\\nGerman:" + target, P pairs per step
               (= per-device batch 4 x gradient accumulation 64), loss on the target only, inverse-sqrt LR with
               warm-up ratio 0.01, weight decay 0.01, LoRA dropout 0.05 (the repo's ALMA scripts)

variants (--variant):
  sft    no modes (K = 1, r_p = 0): LoRA-MLE / sft1
  wta    K modes, hard winner-takes-all of wta1: no-grad loss of every mode on every example, assignment by relative
         loss with capacity 1.25 B / K per mode, a random other mode with probability eps (0.3 -> 0.05 over the first
         20% of training); then one pass with gradient through the assigned mode
  mcl    relaxed WTA of LoRA-MCL (their Eq. 5, eps = --mcl_eps): every example through all K modes with gradient,
         weight 1 - eps for the lowest loss, eps / (K - 1) for the others
  oracle (math, --modc_file) supervised routing: the reference of strategy k always trains mode k (no eps on
         solutions); verifier examples as in wta (by loss, with eps)
  match  (math, --modc_file) wta inside a problem (match_assign): the references of one problem (different strategies)
         always go to different modes, the best such assignment by loss, with the batch capacity of wta1; exploration
         replaces a whole problem's assignment by a random one with distinct modes (probability eps), so the
         constraint holds at every step; problems in random order, ties broken at random (the references are sorted
         by strategy: a fixed tie order would leak the labels)
The private share is --r_p / (--r_s + --r_p); FiLM codes with --film 1.

--modc_file (math): rows {id, question, refs: [{mode: strategy, text}]} of phase1/modc_data.py (the first correct
solution of every prompt strategy, <= 4 per problem) instead of the single reference of train.jsonl; sft and wta take
it too (the controls with the same data: sft without modes, wta with batch assignment that may give two strategies of
one problem the same mode). A step holds whole problems, P = round(16 / references per problem) by default, so a step
has about as many solution examples as with one reference.

Budget: --target_tokens (or --target_tokens_from a result.json: its "tokens") and/or --epochs; the run ends at
whichever comes first and the LR schedule follows the same progress. --max_hours is a safety limit only.
Checkpoints every --ckpt_minutes (0: never, the disk is small) with Adam moments; re-running resumes."""
import argparse
import itertools
import json
import math
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase1 import priv as PV  # noqa: E402
from phase1.train import (assign, balanced, collate, load_jsonl, make_steps, mcl_weights, micro,  # noqa: E402
                          per_example_nll)

MT_PROMPT = "Translate this from English to German:\nEnglish: {src}\nGerman:"
MT_MAX_NEW = 256


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- MT examples
def mt_prompt(src):
    return MT_PROMPT.format(src=src)


def encode_mt(tok, src, tgt):
    """(prompt ids with BOS, target ids + EOS, prompt tokens unchanged by the target?). As the ALMA code, prompt and
    target are one string (no space added between "German:" and the target), tokenized together, and the first
    len(tokens of the prompt alone) tokens are the prompt (their loss mask). The prompt is never truncated (the same
    prompt as in generation); the target is cut at MT_MAX_NEW tokens with EOS."""
    p_alone = tok(mt_prompt(src), add_special_tokens=True)["input_ids"]
    full = tok(mt_prompt(src) + tgt, add_special_tokens=True)["input_ids"]
    p = full[:len(p_alone)]
    t = full[len(p_alone):][:MT_MAX_NEW - 1] + [tok.eos_token_id]
    return p, t, p == p_alone


def mt_steps(train, tok, P, seed):
    """One epoch of steps of P pairs (a new order per epoch)."""
    rng = random.Random(seed)
    order = list(range(len(train)))
    rng.shuffle(order)
    steps, bad = [], 0
    for s in range(0, len(order), P):
        ex = []
        for g in order[s:s + P]:
            p, t, ok = encode_mt(tok, train[g]["src"], train[g]["tgt"])
            bad += not ok
            ex.append(dict(p=p, t=t, group=g, kind="solve", mode=None, w=1.0))
        steps.append(ex)
    return steps, bad


# ----------------------------------------------------------------------------- schedules
def lr_mult(sched, step, total_steps, prog, warm_steps):
    """step: optimizer steps done (0-based for the next one); prog in [0, 1]."""
    if sched == "cos":                     # phase 1 (train.py)
        warm = min(1.0, (step + 1) / warm_steps)
        return warm * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))
    # Hugging Face get_inverse_sqrt_schedule (timescale = warm-up steps) at its current_step = optimizer steps done:
    # the first step has LR 0 during the warm-up, as there
    w = warm_steps
    if step < w:
        return step / max(1, w)
    ts = w or 10_000
    return 1.0 / math.sqrt((step + ts - w) / ts)


def eps_at(prog, start, final):
    return start + (final - start) * min(1.0, prog / 0.2)


def match_assign(ex, L, K, eps, rng, cap_factor=1.25):
    """Mode of every example for --variant match. Solutions: the references of one problem get distinct modes. The
    problems are taken in random order; each gets, among the modes still under the capacity ceil(cap_factor * n / K)
    of wta1 (or the least loaded ones when too few are left), the injective assignment with the lowest summed loss
    (permutations tried in random order, so ties fall at random); with probability eps a random injective assignment
    instead. Verifier examples: balanced by relative loss with eps, as in wta1."""
    modes = [0] * len(ex)
    solve = [i for i, e in enumerate(ex) if e["kind"] == "solve"]
    ver = [i for i, e in enumerate(ex) if e["kind"] == "verify"]
    groups = {}
    for i in solve:
        groups.setdefault(ex[i]["group"], []).append(i)
    if any(len(v) > K for v in groups.values()):
        raise ValueError(f"a problem has more references than the {K} modes")
    cap = math.ceil(len(solve) / K * cap_factor)
    load = [0] * K
    order = list(groups)
    rng.shuffle(order)
    for g in order:
        idx = groups[g]
        M = len(idx)
        if rng.random() < eps:
            perm = rng.sample(range(K), M)
        else:
            free = [k for k in range(K) if load[k] < cap]
            pool = free if len(free) >= M else sorted(range(K), key=lambda k: (load[k], k))[:M]
            perms = list(itertools.permutations(pool, M))
            rng.shuffle(perms)
            perm = min(perms, key=lambda p: sum(float(L[p[m], idx[m]]) for m in range(M)))
        for i, k in zip(idx, perm):
            modes[i] = k
            load[k] += 1
    if ver:
        for i, k in zip(ver, balanced(L[:, ver])):
            modes[i] = k
        for i in ver:
            if rng.random() < eps:
                modes[i] = (modes[i] + rng.randrange(1, K)) % K
    return modes


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["math", "mt"])
    ap.add_argument("--variant", required=True, choices=["sft", "wta", "mcl", "oracle", "match"])
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True, help="math: dir with train.jsonl, verify.jsonl; mt: dir with train.jsonl")
    ap.add_argument("--modc_file", default="", help="math: strategy-labelled references (phase1/modc_data.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--r_s", type=int, default=32)
    ap.add_argument("--r_p", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0, help="0: 2 (r_s + r_p), scale 2 as in phase 1 / LoRA-MCL")
    ap.add_argument("--film", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=-1, help="-1: 0 for math, 0.05 for mt")
    ap.add_argument("--lr", type=float, default=0)
    ap.add_argument("--film_lr", type=float, default=0, help="0: 5 x --lr (phase 1: 1e-3 at 2e-4)")
    ap.add_argument("--wd", type=float, default=-1, help="-1: 0 for math, 0.01 for mt")
    ap.add_argument("--P", type=int, default=0, help="problems (math, 16) / pairs (mt, 256) per optimizer step")
    ap.add_argument("--ver_ratio", type=float, default=0.4)
    ap.add_argument("--ver_coef", type=float, default=0.4)
    ap.add_argument("--mb_tokens", type=int, default=0, help="padded tokens per micro-batch (math 6144, mt 4096)")
    ap.add_argument("--sched", default="", choices=["", "cos", "invsqrt"])
    ap.add_argument("--warmup", type=float, default=-1, help="cos: steps (30); invsqrt: ratio of the steps (0.01)")
    ap.add_argument("--target_tokens", type=int, default=0)
    ap.add_argument("--target_tokens_from", default="")
    ap.add_argument("--epochs", type=float, default=0, help="0: no limit by passes (math), mt: 2")
    ap.add_argument("--max_steps", type=int, default=0, help="smoke runs")
    ap.add_argument("--max_hours", type=float, default=12.0, help="safety limit")
    ap.add_argument("--eps_start", type=float, default=0.3)
    ap.add_argument("--eps_final", type=float, default=0.05)
    ap.add_argument("--mcl_eps", type=float, default=0.05)
    ap.add_argument("--ckpt_minutes", type=float, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    mt = a.task == "mt"
    if a.variant in ("oracle", "match") and not a.modc_file:
        raise SystemExit(f"--variant {a.variant} needs --modc_file (references labelled by strategy)")
    if a.modc_file and (mt or a.variant == "mcl"):
        raise SystemExit("--modc_file is for math with sft, wta, oracle, match")
    modc = load_jsonl(a.modc_file) if a.modc_file else None
    n_refs = 1.0
    if modc is not None:
        if not modc or any(not r.get("refs") for r in modc):
            raise SystemExit(f"{a.modc_file}: rows without references")
        n_refs = sum(len(r["refs"]) for r in modc) / len(modc)
        top = max(ref["mode"] for r in modc for ref in r["refs"])
        most = max(len(r["refs"]) for r in modc)
        if a.variant == "oracle" and top >= a.K:
            raise SystemExit(f"oracle: strategy {top} in {a.modc_file} needs --K > {top}")
        if a.variant == "match" and most > a.K:
            raise SystemExit(f"match: a problem has {most} references, more than --K {a.K} modes")
        a.P = a.P or max(1, round(16 / n_refs))
    a.P = a.P or (256 if mt else 16)
    a.lr = a.lr or (2e-5 if mt else 2e-4)
    a.film_lr = a.film_lr or 5 * a.lr
    a.wd = (0.01 if mt else 0.0) if a.wd < 0 else a.wd
    a.dropout = (0.05 if mt else 0.0) if a.dropout < 0 else a.dropout
    a.mb_tokens = a.mb_tokens or (4096 if mt else 6144)
    a.sched = a.sched or ("invsqrt" if mt else "cos")
    a.epochs = a.epochs or (2.0 if mt else 0.0)
    K = 1 if a.variant == "sft" else a.K
    if a.variant == "sft" and (a.r_p or a.film):
        raise SystemExit("sft has no modes: --r_p 0 --film 0")
    if a.variant != "sft" and K < 2:
        raise SystemExit("modes need --K >= 2")
    if a.variant != "sft" and not (a.r_p or a.film):
        raise SystemExit("modes without private LoRA and without FiLM would be identical")
    alpha = a.alpha or 2.0 * (a.r_s + a.r_p)
    os.makedirs(a.out, exist_ok=True)
    target = a.target_tokens or (json.load(open(a.target_tokens_from))["tokens"] if a.target_tokens_from else 0)
    # everything that changes the result: a finished or resumed run with another recipe is refused
    fp = {f: [os.path.getsize(f), int(os.path.getmtime(f))] for f in
          [os.path.join(a.data, "train.jsonl")] + ([os.path.join(a.data, "verify.jsonl")] if not mt else [])
          + ([os.path.abspath(a.modc_file)] if a.modc_file else []) if os.path.exists(f)}
    # modc_file only when given: runs made before this option keep their recipe
    recipe = {k: v for k, v in vars(a).items() if k not in ("max_hours", "ckpt_minutes", "out")
              and not (k == "modc_file" and not v)}
    if a.modc_file:
        recipe["modc_file"] = os.path.abspath(a.modc_file)
    recipe.update(base=os.path.abspath(a.base), data=os.path.abspath(a.data), inputs=fp, target=target, alpha=alpha,
                  target_tokens_from=os.path.abspath(a.target_tokens_from) if a.target_tokens_from else "")
    rf = os.path.join(a.out, "recipe.json")
    if os.path.exists(rf):
        old = json.load(open(rf))
        diff = {k: (old.get(k), v) for k, v in recipe.items() if old.get(k) != v}
        if diff:
            raise SystemExit(f"{a.out} was trained with another recipe (was, now): {diff}; use another --out")
    else:
        json.dump(recipe, open(rf, "w"), indent=1)
    if os.path.exists(os.path.join(a.out, "DONE")):
        return
    from transformers import AutoTokenizer
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.base)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    cfg = {"format": "priv", "task": a.task, "variant": a.variant, "K": K, "r_s": a.r_s, "r_p": a.r_p,
           "alpha": alpha, "film": bool(a.film), "dropout": a.dropout, "seed": a.seed}
    if a.modc_file:
        cfg["refs"] = "modc"
    model, film, ctx = PV.build(a.base, K, a.r_s, a.r_p, alpha, bool(a.film), dev, True, a.dropout)
    lora_ps, film_ps = PV.trainable(model, film)
    groups = [{"params": lora_ps, "lr": a.lr}] + ([{"params": film_ps, "lr": a.film_lr}] if film_ps else [])
    opt = torch.optim.AdamW(groups, weight_decay=a.wd, betas=(0.9, 0.99) if not mt else (0.9, 0.999))
    train = modc if modc is not None else load_jsonl(os.path.join(a.data, "train.jsonl"))
    vf = os.path.join(a.data, "verify.jsonl")
    verify = load_jsonl(vf) if not mt and os.path.exists(vf) else []

    def epoch_steps(e):
        if mt:
            st_, bad = mt_steps(train, tok, a.P, a.seed + 1000 * e)
            if bad and e == 0:
                log(f"{bad} pairs whose prompt tokens change when the target follows (split at the prompt length)")
            return st_
        # modc rows: every reference of a problem, its strategy in "mode", no prefix in the prompt
        return make_steps("sftmodc" if modc is not None else "wta1", train, verify, tok, a.P, a.ver_ratio,
                          a.seed + 1000 * e, max(K, 1))

    cache = {0: epoch_steps(0)}
    per_epoch = len(cache[0])

    def get_step(i):
        e = i // per_epoch
        if e not in cache:
            cache.clear()
            cache[e] = epoch_steps(e)
        return cache[e][i % per_epoch]

    total = math.ceil(a.epochs * per_epoch) if a.epochs else 0
    if a.max_steps:
        total = min(total, a.max_steps) if total else a.max_steps
    if not (target or total):
        raise SystemExit("no budget: give --target_tokens(_from), --epochs or --max_steps")
    plan_steps = total or max(1, round(target / max(1, sum(len(e["p"]) + len(e["t"]) for s in cache[0] for e in s)
                                                    / per_epoch)))
    warm = (a.warmup if a.warmup >= 0 else 30) if a.sched == "cos" else \
        math.ceil((a.warmup if a.warmup >= 0 else 0.01) * plan_steps)         # HF TrainingArguments.get_warmup_steps
    st = {"step": 0, "tokens": 0, "elapsed": 0.0}
    share = torch.zeros(K)
    rng = random.Random(a.seed)
    last = os.path.join(a.out, "last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=False)
        if ck["cfg"] != cfg or ck.get("recipe") != recipe:
            raise SystemExit(f"{last} has another configuration: {ck['cfg']} vs {cfg}")
        PV.load_state(model, film, ck["adapter"])
        opt.load_state_dict(ck["opt"])
        st, share = ck["state"], ck["share"]
        rng.setstate(ck["rng"])
        torch.set_rng_state(ck["torch_rng"])             # dropout masks continue as without the stop
        if dev == "cuda" and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng"])
        log(f"resumed at step {st['step']}")
        del ck
    n_tr = sum(p.numel() for p in lora_ps + film_ps)
    log(f"{a.task} {a.variant}: K {K}, r_s {a.r_s}, r_p {a.r_p} (private {a.r_p / (a.r_s + a.r_p):.0%}), alpha {alpha:g}, "
        f"FiLM {bool(a.film)}, trainable {n_tr / 1e6:.1f}M, {per_epoch} steps/epoch, budget: tokens {target or '-'}, "
        f"steps {total or '-'} (planned {plan_steps}), LR {a.lr:g} ({a.sched}, warm-up {warm})")
    t0, t_ck = time.time() - st["elapsed"], time.time()
    agg = {"loss": 0.0, "n": 0}
    model.train()
    stop = ""

    def save_last():
        torch.save({"adapter": PV.adapter_state(model, film), "opt": opt.state_dict(), "state": st, "cfg": cfg,
                    "recipe": recipe, "rng": rng.getstate(), "share": share, "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state() if dev == "cuda" else None}, last + ".tmp")
        os.replace(last + ".tmp", last)

    while True:
        prog = max(st["tokens"] / target if target else 0.0, st["step"] / total if total else 0.0)
        if prog >= 1:
            stop = "budget"
            break
        if time.time() - t0 > a.max_hours * 3600:
            save_last()                    # an undertrained point is not finished: no DONE, re-running resumes
            raise SystemExit(f"safety time limit ({a.max_hours} h) reached at {st['tokens']} tokens, step "
                             f"{st['step']}, before the budget; {last} saved, re-run with a larger --max_hours")
        mult = lr_mult(a.sched, st["step"], plan_steps, prog, warm)
        for g, lr0 in zip(opt.param_groups, [a.lr, a.film_lr]):
            g["lr"] = lr0 * mult
        ex = get_step(st["step"])
        eps = eps_at(prog, a.eps_start, a.eps_final)
        modes = None
        if a.variant in ("wta", "oracle", "match"):
            # the loss of every mode decides the assignment; oracle needs it for the verifier examples only
            need = [i for i, e in enumerate(ex) if a.variant != "oracle" or e["kind"] == "verify"]
            sub = [ex[i] for i in need]
            L = torch.zeros(K, len(ex))
            model.eval()                   # without dropout noise
            with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                for ids in micro(sub, a.mb_tokens):
                    x, att, lab = collate(sub, ids, pad, dev)
                    cols = [need[j] for j in ids]
                    for k in range(K):
                        md = torch.full((len(ids),), k, dtype=torch.long, device=dev)
                        L[k, cols] = per_example_nll(model, film, ctx, x, att, lab, md).float().cpu()
            model.train()
            # wta: train.assign wta1 (batch-balanced by relative loss); oracle: train.assign fixed (the mode the
            # example carries, here its strategy); match: distinct modes inside a problem; verifiers balanced in all
            if a.variant == "match":
                modes = match_assign(ex, L, K, eps, rng)
            else:
                modes = assign({"wta": "wta1", "oracle": "fixed"}[a.variant], ex, L, K, eps, rng)
            share += torch.bincount(torch.tensor([modes[i] for i, e in enumerate(ex) if e["kind"] == "solve"]),
                                    minlength=K).float()
        w_solve = sum(e["w"] for e in ex if e["kind"] == "solve")
        n_ver = sum(e["kind"] == "verify" for e in ex)
        for ids in micro(ex, a.mb_tokens // K if a.variant == "mcl" else a.mb_tokens):
            x, att, lab = collate(ex, ids, pad, dev)
            w = torch.tensor([ex[i]["w"] / max(w_solve, 1e-9) if ex[i]["kind"] == "solve" else a.ver_coef / n_ver
                              for i in ids], device=dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                if a.variant == "mcl":
                    n = len(ids)
                    md = torch.arange(K, device=dev).repeat(n)
                    nll = per_example_nll(model, film, ctx, x.repeat_interleave(K, 0), att.repeat_interleave(K, 0),
                                          lab.repeat_interleave(K, 0), md).view(n, K)
                    ((nll * mcl_weights(nll.detach(), a.mcl_eps)).sum(1) * w).sum().backward()
                    nll, win = nll.detach().min(1)
                    for i, k in zip(ids, win.tolist()):
                        share[k] += ex[i]["kind"] == "solve"
                else:
                    md = torch.tensor([modes[i] for i in ids], device=dev) if modes is not None else None
                    nll = per_example_nll(model, film, ctx, x, att, lab, md)
                    (nll * w).sum().backward()
            for i, v in zip(ids, nll.detach().float().cpu().tolist()):
                if ex[i]["kind"] == "solve":
                    agg["loss"] += v
                    agg["n"] += 1
            st["tokens"] += int(att.sum())
        torch.nn.utils.clip_grad_norm_(lora_ps + film_ps, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        st["step"] += 1
        st["elapsed"] = time.time() - t0
        if st["step"] % 10 == 0 or st["step"] == 1:
            log(f"step {st['step']} (epoch {st['step'] / per_epoch:.2f}, progress {prog:.2f}) loss "
                f"{agg['loss'] / max(agg['n'], 1):.4f} | {st['tokens'] / max(st['elapsed'], 1):.0f} tok/s | "
                f"{st['elapsed'] / 3600:.2f} h | LR x{mult:.3f}"
                + (f" | modes {[round(v, 2) for v in (share / share.sum()).tolist()]}" if share.sum() else ""))
            agg = {"loss": 0.0, "n": 0}
        if a.ckpt_minutes > 0 and time.time() - t_ck > a.ckpt_minutes * 60:
            save_last()
            t_ck = time.time()
    torch.save({"adapter": PV.adapter_state(model, film), "cfg": cfg}, os.path.join(a.out, "adapter.tmp"))
    os.replace(os.path.join(a.out, "adapter.tmp"), os.path.join(a.out, "adapter.pt"))
    res = {"task": a.task, "variant": a.variant, "K": K, "r_s": a.r_s, "r_p": a.r_p, "private_share": a.r_p / (a.r_s + a.r_p),
           "film": bool(a.film), "alpha": alpha, "steps_done": st["step"], "steps_per_epoch": per_epoch,
           "epochs_done": st["step"] / per_epoch, "tokens": st["tokens"], "target_tokens": target, "hours": st["elapsed"] / 3600,
           "trainable_M": n_tr / 1e6, "seed": a.seed, "stop": stop, "refs": "modc" if modc is not None else "single",
           "P": a.P, "refs_per_problem": n_refs if modc is not None else 1.0,
           "mode_share": (share / share.sum()).tolist() if share.sum() else None}
    json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    if os.path.exists(last):
        os.remove(last)
    open(os.path.join(a.out, "DONE"), "w").close()
    log("done", res)


if __name__ == "__main__":
    main()
