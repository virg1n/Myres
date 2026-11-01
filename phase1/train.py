"""Phase-1 training of LoRA (+ FiLM modes) on Qwen3.

variants (--variant):
  sft1       no modes, one reference per problem
  sftmulti   no modes, all structurally distinct references (<= K)
  filmmean   K modes, multi-ref, random mode per example (z without WTA)
  wta1       K modes, one reference, relaxed WTA with per-batch capacity balancing
  wtamulti   K modes, multi-ref, each reference goes to the mode with the lowest loss on it
  wtabal     K modes, multi-ref, references of one problem matched to distinct modes by
             relative loss (optimal assignment; never forces more splits than distinct refs)
  fixed      examples carry their mode and weight (stage-2 rounds)
  loramcl    LoRA-MCL: K separate adapters, no FiLM, one reference, relaxed WTA (Eq. 5 of the paper):
             every example goes through all K adapters with weight 1-eps for the winner, eps/(K-1) else
  modc       ModC with prefixes: one adapter, "[Mode k]" in the input, reference of strategy k trains mode k
Verification examples (Yes/No) are mixed in for every variant; with modes they use batch WTA
(loramcl: relaxed WTA as for solutions; modc: a random prefix).
Training stops at the end of one pass or at --max_hours, the LR schedule follows whichever comes
first, and a checkpoint is written every --ckpt_minutes (re-running resumes).
--snap_tokens T saves the adapter once T training tokens are reached (optionally stopping there)."""
import argparse
import itertools
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase0 import prompts  # noqa: E402
from phase1 import film as FM  # noqa: E402

MAX_TARGET = 1300
MODE_VARIANTS = {"filmmean", "wta1", "wtamulti", "wtabal"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


# ----------------------------------------------------------------------------- examples
def encode_solve(tok, q, sol, prefix=None):
    p = tok.encode(prompts.solve_prompt(tok, q, "neutral", prefix), add_special_tokens=False)
    t = tok.encode(sol, add_special_tokens=False)[:MAX_TARGET] + tok.encode("<|im_end|>", add_special_tokens=False)
    return p, t


def encode_verify(tok, q, sol, label, prefix=None):
    p = tok.encode(prompts.verify_prompt(tok, q, sol, "neutral", prefix), add_special_tokens=False)[-2400:]
    t = tok.encode(label + "<|im_end|>", add_special_tokens=False)
    return p, t


def make_steps(variant, train, verify, tok, P, ver_ratio, seed, K=4):
    """List of steps; each step = list of examples dict(p, t, group, kind, mode, w)."""
    rng = random.Random(seed)
    order = list(range(len(train)))
    rng.shuffle(order)
    vi = 0
    steps = []
    for s in range(0, len(order), P):
        ex = []
        for g in order[s:s + P]:
            r = train[g]
            if variant == "fixed":
                p, t = encode_solve(tok, r["question"], r["solution"])
                ex.append(dict(p=p, t=t, group=g, kind="solve", mode=r.get("mode"), w=r.get("w", 1.0)))
                continue
            if variant == "modc":
                for ref in r["refs"]:
                    p, t = encode_solve(tok, r["question"], ref["text"], ref["mode"])
                    ex.append(dict(p=p, t=t, group=g, kind="solve", mode=ref["mode"], w=1.0))
                continue
            refs = [r["single"]] if variant in ("sft1", "wta1", "loramcl") else r["refs"]
            for sol in refs:
                p, t = encode_solve(tok, r["question"], sol)
                ex.append(dict(p=p, t=t, group=g, kind="solve", mode=None, w=1.0))
        n_ver = round(len(ex) * ver_ratio)
        for _ in range(n_ver):
            if not verify:
                break
            v = verify[vi % len(verify)]
            vi += 1
            prefix = rng.randrange(K) if variant == "modc" else None
            p, t = encode_verify(tok, v["question"], v["solution"], v["label"], prefix)
            ex.append(dict(p=p, t=t, group=-1, kind="verify", mode=None, w=1.0))
        steps.append(ex)
    return steps


def micro(ex, budget):
    """Split examples into micro-batches of at most `budget` padded tokens (sorted by length)."""
    idx = sorted(range(len(ex)), key=lambda i: len(ex[i]["p"]) + len(ex[i]["t"]))
    out, cur, mx = [], [], 0
    for i in idx:
        L = len(ex[i]["p"]) + len(ex[i]["t"])
        if cur and max(mx, L) * (len(cur) + 1) > budget:
            out.append(cur)
            cur, mx = [], 0
        cur.append(i)
        mx = max(mx, L)
    if cur:
        out.append(cur)
    return out


def collate(ex, ids, pad, dev):
    T = max(len(ex[i]["p"]) + len(ex[i]["t"]) for i in ids)
    x = torch.full((len(ids), T), pad, dtype=torch.long)
    att = torch.zeros((len(ids), T), dtype=torch.long)
    lab = torch.full((len(ids), T), -100, dtype=torch.long)
    for r, i in enumerate(ids):
        seq = ex[i]["p"] + ex[i]["t"]
        x[r, :len(seq)] = torch.tensor(seq)
        att[r, :len(seq)] = 1
        lab[r, len(ex[i]["p"]):len(seq)] = torch.tensor(ex[i]["t"])
    return x.to(dev), att.to(dev), lab.to(dev)


def per_example_nll(model, film, ctx, x, att, lab, modes):
    ctx.gamma = film.gamma(modes) if (film is not None and modes is not None) else None
    ctx.modes = modes
    h = model.model(input_ids=x, attention_mask=att).last_hidden_state
    tgt = lab[:, 1:]
    m = tgt != -100
    hs = h[:, :-1][m]
    ce = F.cross_entropy(model.lm_head(hs).float(), tgt[m], reduction="none")
    rows = torch.arange(x.shape[0], device=x.device).unsqueeze(1).expand_as(tgt)[m]
    tot = torch.zeros(x.shape[0], device=x.device).index_add_(0, rows, ce)
    # ctx stays set: gradient checkpointing re-runs the FiLM norms and adapters during backward
    return tot / m.sum(1).clamp(min=1)


def mcl_weights(nll, eps):
    """nll [B, K] -> q [B, K]: relaxed WTA, 1-eps for the lowest loss, eps/(K-1) for the others."""
    K = nll.shape[1]
    q = torch.full_like(nll, eps / (K - 1))
    return q.scatter(1, nll.argmin(1, keepdim=True), 1 - eps)


# ----------------------------------------------------------------------------- assignment
def balanced(L, cap_factor=1.25):
    """L [K, B] -> mode per example, at most cap_factor*B/K per mode, by relative loss."""
    K, B = L.shape
    cap = math.ceil(B / K * cap_factor)
    rel = (L - L.mean(0, keepdim=True)).t().flatten()
    out, load, n = [-1] * B, [0] * K, 0
    for idx in torch.argsort(rel).tolist():
        i, k = divmod(idx, K)
        if out[i] < 0 and load[k] < cap:
            out[i], load[k], n = k, load[k] + 1, n + 1
            if n == B:
                break
    return out


def match(Lg):
    """Lg [K, M] (M <= K) relative losses of one problem -> distinct mode per reference."""
    K, M = Lg.shape
    rel = Lg - Lg.mean(0, keepdim=True)
    best, best_c = None, None
    for perm in itertools.permutations(range(K), M):
        c = sum(float(rel[perm[m], m]) for m in range(M))
        if best_c is None or c < best_c:
            best, best_c = perm, c
    return list(best)


def assign(variant, ex, L, K, eps, rng):
    """Mode for every example of a step. L [K, n_ex] no-grad losses (None if not needed)."""
    modes = [0] * len(ex)
    solve = [i for i, e in enumerate(ex) if e["kind"] == "solve"]
    ver = [i for i, e in enumerate(ex) if e["kind"] == "verify"]
    if variant == "filmmean":
        return [rng.randrange(K) for _ in ex]
    if variant == "fixed":
        for i in solve:
            modes[i] = ex[i]["mode"]
    elif variant == "wta1":
        for i, k in zip(solve, balanced(L[:, solve])):
            modes[i] = k
    elif variant == "wtamulti":
        for i in solve:
            modes[i] = int(L[:, i].argmin())
    elif variant == "wtabal":
        groups = {}
        for i in solve:
            groups.setdefault(ex[i]["group"], []).append(i)
        for g, idx in groups.items():
            for i, k in zip(idx, match(L[:, idx])):
                modes[i] = k
    if ver:
        for i, k in zip(ver, balanced(L[:, ver])):
            modes[i] = k
    for i in range(len(ex)):
        if variant != "fixed" or ex[i]["kind"] == "verify":
            if rng.random() < eps:
                modes[i] = (modes[i] + rng.randrange(1, K)) % K
    return modes


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["sft1", "sftmulti", "filmmean", "wta1", "wtamulti",
                                                          "wtabal", "fixed", "loramcl", "modc"])
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True, help="dir with train.jsonl / verify.jsonl (or --train_file)")
    ap.add_argument("--train_file", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default="", help="adapter.pt to continue from (stage 2)")
    ap.add_argument("--film_modes", type=int, default=-1, help="-1: from variant (K for mode variants)")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--film_lr", type=float, default=1e-3)
    ap.add_argument("--P", type=int, default=16, help="problems per optimizer step")
    ap.add_argument("--ver_ratio", type=float, default=0.4, help="verify examples per solve example")
    ap.add_argument("--mb_tokens", type=int, default=6144)
    ap.add_argument("--max_hours", type=float, default=2.0)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1000, help="max passes; normally the time limit ends training")
    ap.add_argument("--ver_coef", type=float, default=0.4, help="loss = solve loss + ver_coef * verify loss")
    ap.add_argument("--eps_start", type=float, default=0.3)
    ap.add_argument("--eps_final", type=float, default=0.05)
    ap.add_argument("--mcl_eps", type=float, default=0.05, help="loramcl: relaxation of WTA (paper, Table 1)")
    ap.add_argument("--ckpt_minutes", type=float, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_opt", type=int, default=1, help="0: adapter.pt without Adam moments (3x smaller)")
    ap.add_argument("--snap_tokens", type=int, default=0, help="save snap/adapter.pt at this many training tokens")
    ap.add_argument("--stop_after_snap", action="store_true", help="stop at --snap_tokens, adapter.pt is the snapshot")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if os.path.exists(os.path.join(a.out, "DONE")):
        return
    from transformers import AutoTokenizer
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.base)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    init = torch.load(a.init, map_location="cpu", weights_only=False) if a.init else None
    film_modes = a.film_modes if a.film_modes >= 0 else (
        init["cfg"]["film_modes"] if init else (a.K if a.variant in MODE_VARIANTS else 0))
    K = film_modes or (a.K if a.variant in ("loramcl", "modc") else 1)
    mcl = a.variant == "loramcl"
    model, film, ctx = FM.build(a.base, film_modes, a.lora_r, 2 * a.lora_r, dev, lora_modes=K if mcl else 0)
    if init:
        FM.load_adapter(model, film, init["adapter"])
    lora_ps, film_ps = FM.trainable(model, film)
    groups = [{"params": lora_ps, "lr": a.lr}] + ([{"params": film_ps, "lr": a.film_lr}] if film_ps else [])
    opt = torch.optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.99))
    if init and init.get("opt"):
        try:                       # keep Adam moments from the previous stage instead of a cold restart
            opt.load_state_dict(init["opt"])
        except Exception as e:
            log("optimizer state not loaded:", repr(e)[:200])
    train = load_jsonl(a.train_file or os.path.join(a.data, "train.jsonl"))
    verify = load_jsonl(os.path.join(a.data, "verify.jsonl")) if os.path.exists(os.path.join(a.data, "verify.jsonl")) else []
    # every variant trains for the same wall-clock budget: the data is repeated (new order each epoch)
    epoch_cache = {}

    def get_step(i):
        e = i // per_epoch
        if e not in epoch_cache:
            epoch_cache.clear()
            epoch_cache[e] = make_steps(a.variant, train, verify, tok, a.P, a.ver_ratio, a.seed + 1000 * e, K)
        return epoch_cache[e][i % per_epoch]

    first = make_steps(a.variant, train, verify, tok, a.P, a.ver_ratio, a.seed, K)
    per_epoch = max(1, len(first))
    epoch_cache[0] = first
    total = per_epoch * a.epochs
    if a.max_steps:
        total = min(total, a.max_steps)
    st = {"step": 0, "tokens": 0, "elapsed": 0.0}
    last = os.path.join(a.out, "last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=False)
        FM.load_adapter(model, film, ck["adapter"])
        opt.load_state_dict(ck["opt"])
        st = ck["state"]
        log(f"resumed at step {st['step']}")
    cfg = {"variant": a.variant, "film_modes": film_modes, "lora_r": a.lora_r, "lora_alpha": 2 * a.lora_r,
           "lora_modes": K if mcl else 0, "prefix_modes": K if a.variant == "modc" else 0}
    rng = random.Random(a.seed + st["step"])
    n_tr = sum(p.numel() for p in lora_ps + film_ps)
    log(f"{a.variant}: {per_epoch} steps/epoch, {sum(len(s) for s in first)} examples/epoch, trainable {n_tr / 1e6:.1f}M, "
        f"modes {K}, limit {a.max_hours} h")
    t0, t_ck = time.time() - st["elapsed"], time.time()
    share = torch.zeros(K)
    agg = {"loss_s": 0.0, "n_s": 0, "loss_v": 0.0, "n_v": 0}
    model.train()
    while st["step"] < total and first and first[0]:
        el = time.time() - t0
        # LR / eps schedule follow the time budget (or max_steps in smoke runs)
        prog = max(el / (a.max_hours * 3600), st["step"] / a.max_steps if a.max_steps else 0.0)
        if prog >= 1:
            log("time limit reached")
            break
        warm = min(1.0, (st["step"] + 1) / 30)
        mult = warm * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))
        for g, base_lr in zip(opt.param_groups, [a.lr, a.film_lr]):
            g["lr"] = base_lr * mult
        ex = get_step(st["step"])
        mbs = micro(ex, a.mb_tokens // K if mcl else a.mb_tokens)        # loramcl: every row runs K times
        L = None
        if film is not None and a.variant in ("wta1", "wtamulti", "wtabal", "fixed"):
            # no-grad loss of every mode on every example that needs an assignment
            need = [i for i, e in enumerate(ex) if a.variant != "fixed" or e["kind"] == "verify"]
            sub = [ex[i] for i in need]
            L = torch.zeros(K, len(ex))
            with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                for ids in micro(sub, a.mb_tokens):
                    x, att, lab = collate(sub, ids, pad, dev)
                    cols = [need[j] for j in ids]
                    for k in range(K):
                        md = torch.full((len(ids),), k, dtype=torch.long, device=dev)
                        L[k, cols] = per_example_nll(model, film, ctx, x, att, lab, md).float().cpu()
        eps = a.eps_start + (a.eps_final - a.eps_start) * min(1.0, prog / 0.2)
        modes = assign(a.variant, ex, L, K, eps, rng) if film is not None else None
        w_solve = sum(e["w"] for e in ex if e["kind"] == "solve")
        n_ver = sum(e["kind"] == "verify" for e in ex)
        for ids in mbs:
            x, att, lab = collate(ex, ids, pad, dev)
            md = torch.tensor([modes[i] for i in ids], device=dev) if modes is not None else None
            # weight of each example in  sum_solve(w*nll)/sum_solve(w) + ver_coef * mean_verify(nll)
            w = torch.tensor([ex[i]["w"] / max(w_solve, 1e-9) if ex[i]["kind"] == "solve" else a.ver_coef / n_ver
                              for i in ids], device=dev)
            if mcl:
                n = len(ids)
                md = torch.arange(K, device=dev).repeat(n)
                with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                    nll = per_example_nll(model, film, ctx, x.repeat_interleave(K, 0), att.repeat_interleave(K, 0),
                                          lab.repeat_interleave(K, 0), md).view(n, K)
                ((nll * mcl_weights(nll.detach(), a.mcl_eps)).sum(1) * w).sum().backward()
                nll, win = nll.detach().min(1)
                for i, k in zip(ids, win.tolist()):
                    share[k] += ex[i]["kind"] == "solve"
            else:
                with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                    nll = per_example_nll(model, film, ctx, x, att, lab, md)
                (nll * w).sum().backward()
            for r, (i, v) in enumerate(zip(ids, nll.detach().float().cpu().tolist())):
                key = "s" if ex[i]["kind"] == "solve" else "v"
                agg["loss_" + key] += v
                agg["n_" + key] += 1
            st["tokens"] += int(att.sum())
        torch.nn.utils.clip_grad_norm_(lora_ps + film_ps, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if modes is not None:
            share += torch.bincount(torch.tensor([modes[i] for i, e in enumerate(ex) if e["kind"] == "solve"]),
                                    minlength=K).float()
        elif a.variant == "modc":
            for e in ex:
                if e["kind"] == "solve":
                    share[e["mode"]] += 1
        st["step"] += 1
        st["elapsed"] = time.time() - t0
        if a.snap_tokens and not st.get("snapped") and st["tokens"] >= a.snap_tokens:
            snap = os.path.join(a.out, "snap")
            os.makedirs(snap, exist_ok=True)
            if not a.stop_after_snap:
                torch.save({"adapter": FM.adapter_state(model, film), "cfg": cfg}, os.path.join(snap, "adapter.tmp"))
                os.replace(os.path.join(snap, "adapter.tmp"), os.path.join(snap, "adapter.pt"))
            json.dump({"step": st["step"], "epochs": st["step"] / per_epoch, "tokens": st["tokens"],
                       "hours": st["elapsed"] / 3600, "prog": prog, "lr_mult": mult},
                      open(os.path.join(snap, "snap.json"), "w"), indent=1)
            st["snapped"] = True
            log(f"snapshot at {st['tokens'] / 1e6:.1f}M tokens, step {st['step']}, LR x{mult:.2f}")
            if a.stop_after_snap:
                break
        if st["step"] % 10 == 0 or st["step"] == total:
            tps = st["tokens"] / max(st["elapsed"], 1)
            log(f"step {st['step']} (epoch {st['step'] / per_epoch:.2f}) loss solve {agg['loss_s'] / max(agg['n_s'], 1):.4f} verify "
                f"{agg['loss_v'] / max(agg['n_v'], 1):.4f} | {tps:.0f} tok/s | {st['elapsed'] / 3600:.2f} h"
                + (f" | modes {[round(x, 2) for x in (share / share.sum().clamp(min=1)).tolist()]}" if share.sum() else ""))
            agg = {"loss_s": 0.0, "n_s": 0, "loss_v": 0.0, "n_v": 0}
        if time.time() - t_ck > a.ckpt_minutes * 60:
            torch.save({"adapter": FM.adapter_state(model, film), "opt": opt.state_dict(), "state": st, "cfg": cfg},
                       last + ".tmp")
            os.replace(last + ".tmp", last)
            t_ck = time.time()
    torch.save({"adapter": FM.adapter_state(model, film), "cfg": cfg,
                **({"opt": opt.state_dict()} if a.save_opt else {})}, os.path.join(a.out, "adapter.pt"))
    res = {"variant": a.variant, "steps_done": st["step"], "steps_per_epoch": per_epoch,
           "epochs_done": st["step"] / per_epoch, "time_limit_h": a.max_hours, "tokens": st["tokens"],
           "hours": st["elapsed"] / 3600, "trainable_M": n_tr / 1e6, "seed": a.seed,
           "mode_share": (share / share.sum().clamp(min=1)).tolist() if share.sum() else None}
    json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    if os.path.exists(last):
        os.remove(last)
    open(os.path.join(a.out, "DONE"), "w").close()
    log("done", res)


if __name__ == "__main__":
    main()
