"""Stage 1: teacher-forced training from scratch.

  --method base  --z none   baseline (A, A')
  --method wta   --z fix    4 fixed codes + relaxed WTA (C-fix)
  --method wta   --z cont   continuous z, K samples per example + relaxed WTA (C-cont)
  --method mean  --z fix    random code per example, ordinary loss (D-fix)
  --method mean  --z cont   random z per example, ordinary loss (D-cont)

Relaxed WTA is implemented stochastically: each example trains its winner z with prob 1-eps,
otherwise a uniformly random loser (same expected gradient as weights 1-eps / eps/(K-1),
but only one forward+backward). eps is annealed from eps_start to eps_final.

For fixed codes the winners are assigned with a per-batch capacity (balance * B/K examples per
code, by relative loss): plain argmin lets one code win everything early on ("dead" modes).
"""
import argparse
import csv
import math
import os
import time

import torch

from . import igsm
from .common import (autocast, cli_main, device, is_done, log, mark_done, mode_z, model_cfg,
                     save_model, set_seed, write_json)
from .data import train_loader
from .model import GPT, per_example_nll


def make_optimizer(model, lr, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (decay if p.dim() >= 2 and n != "codebook" else no_decay).append(p)
    kw = {"fused": True} if torch.cuda.is_available() else {}
    return torch.optim.AdamW([{"params": decay, "weight_decay": wd},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=lr, betas=(0.9, 0.95), **kw)


def lr_at(frac, lr, warmup=0.02, final=0.1):
    if frac < warmup:
        return lr * frac / warmup
    p = min(1.0, (frac - warmup) / (1 - warmup))
    return lr * (final + (1 - final) * 0.5 * (1 + math.cos(math.pi * p)))


def eps_at(frac, start, final, anneal):
    if anneal <= 0:
        return final
    return start + (final - start) * min(1.0, frac / anneal)


def balanced_assign(L, cap_factor):
    """Assign each example to a mode by relative loss, at most cap_factor*B/K examples per mode."""
    B, K = L.shape
    if cap_factor <= 0:
        return L.argmin(1)
    cap = math.ceil(B / K * cap_factor)
    rel = (L - L.mean(1, keepdim=True)).flatten()
    assigned, load, n = [-1] * B, [0] * K, 0
    for idx in torch.argsort(rel).tolist():
        i, k = divmod(idx, K)
        if assigned[i] < 0 and load[k] < cap:
            assigned[i] = k
            load[k] += 1
            n += 1
            if n == B:
                break
    return torch.tensor(assigned, device=L.device)


def choose_z(fmodel, model, x, y, z_mode, method, K, eps, balance=0.0):
    """z for the training forward pass and stats.

    fix: codes chosen per example (gradient flows into the codebook); cont: vectors."""
    B, dev = x.shape[0], x.device
    if z_mode == "none":
        return None, None
    if method == "mean":
        if z_mode == "fix":
            codes = torch.randint(0, K, (B,), device=dev)
            return model.code_z(codes), codes
        return torch.randn(B, model.cfg.zdim, device=dev), None
    with torch.no_grad():
        if z_mode == "fix":
            Z = model.codebook[:K].detach().unsqueeze(0).expand(B, K, -1)
        else:
            Z = torch.randn(B, K, model.cfg.zdim, device=dev)
        L = torch.stack([per_example_nll(fmodel(x, Z[:, k].contiguous()), y) for k in range(K)], 1)
        winner = balanced_assign(L, balance) if z_mode == "fix" else L.argmin(1)
        loser = (winner + torch.randint(1, K, (B,), device=dev)) % K
        target = torch.where(torch.rand(B, device=dev) < eps, loser, winner)
    if z_mode == "fix":
        return model.code_z(target), winner
    return Z[torch.arange(B, device=dev), target], winner


def maybe_compile(model, enabled, dev, z_mode):
    if not enabled or dev.type != "cuda":
        return model
    try:
        cm = torch.compile(model, dynamic=True)
        x = torch.randint(0, model.cfg.vocab, (4, 128), device=dev)
        z = None if z_mode == "none" else torch.randn(4, model.cfg.zdim, device=dev)
        with autocast(dev):
            with torch.no_grad():
                cm(x, z)
            cm(x, z).float().mean().backward()
        model.zero_grad(set_to_none=True)
        log("torch.compile: ok")
        return cm
    except Exception as e:  # never lose a night to a compiler problem
        log("torch.compile failed, falling back to eager:", repr(e)[:300])
        model.zero_grad(set_to_none=True)
        return model


@torch.no_grad()
def quick_eval(model, z_mode, op_lo, op_hi, n, seed=1234, modes=4, batch_size=512):
    """Greedy accuracy per mode on a fixed val set."""
    probs = igsm.fixed_set(seed, "val", n, op_lo, op_hi, model.cfg.max_len)
    prompts = [igsm.solve_prompt(p) for p in probs]
    model.eval()
    accs = []
    t0 = time.time()
    n_modes = 1 if z_mode == "none" else modes
    g = torch.Generator(device=model.emb.weight.device)
    g.manual_seed(seed)
    for k in range(n_modes):
        z = mode_z(model, z_mode, k, len(prompts), g)
        outs = model.generate(prompts, z=z, temperature=0.0, eos_id=igsm.EOS, batch_size=batch_size)
        accs.append(sum(igsm.check_solution(p, o)["correct"] for p, o in zip(probs, outs)) / n)
    gps = n * n_modes / (time.time() - t0)
    model.train()
    return accs, gps


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", choices=["base", "wta", "mean"], default="base")
    ap.add_argument("--z", choices=["none", "fix", "cont"], default="none")
    ap.add_argument("--tokens", type=float, required=True)
    ap.add_argument("--op_lo", type=int, default=2)
    ap.add_argument("--op_max", type=int, required=True)
    ap.add_argument("--size", default="20m")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--eps_start", type=float, default=0.5)
    ap.add_argument("--eps_final", type=float, default=0.05)
    ap.add_argument("--eps_anneal", type=float, default=0.2)
    ap.add_argument("--balance", type=float, default=1.25, help="per-code capacity factor (fix WTA); 0=off")
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--micro_bs", type=int, default=96)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--verify_frac", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--early_frac", type=float, default=0.0, help="save early.pt at this fraction")
    ap.add_argument("--ckpt_minutes", type=float, default=20)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--quick_eval_n", type=int, default=300)
    ap.add_argument("--bench_steps", type=int, default=0, help=">0: only measure throughput")
    return ap.parse_args()


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    if is_done(a.out):
        log("already done:", a.out)
        return
    dev = device()
    set_seed(a.seed)
    model = GPT(model_cfg(a.size)).to(dev)
    opt = make_optimizer(model, a.lr, a.wd)
    T = int(a.tokens)
    st = {"tokens": 0, "step": 0, "early_saved": False}
    last = os.path.join(a.out, "last.pt")
    if os.path.exists(last) and not a.bench_steps:
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        st = ck["state"]
        log(f"resumed at step {st['step']} tokens {st['tokens'] / 1e9:.3f}B")
    meta = {"z": a.z, "method": a.method, "op_lo": a.op_lo, "op_max": a.op_max, "K": a.K,
            "size": a.size, "tokens": T}
    log(f"{a.out}: method={a.method} z={a.z} tokens={T / 1e9:.2f}B trunk params={model.n_params() / 1e6:.2f}M")
    fmodel = maybe_compile(model, a.compile, dev, a.z)
    loader = train_loader(a.seed * 100003 + st["step"], a.op_lo, a.op_max, a.verify_frac, a.micro_bs,
                          a.workers, model.cfg.max_len)
    logf = open(os.path.join(a.out, "log.csv"), "a", newline="")
    w = csv.writer(logf)
    if st["step"] == 0:
        w.writerow(["step", "tokens", "lr", "eps", "loss_solve", "loss_verify", "tok_per_s"] +
                   [f"win{k}" for k in range(a.K)])
    model.train()
    agg = {"s": 0.0, "ns": 0, "v": 0.0, "nv": 0, "win": torch.zeros(a.K)}
    win_total = torch.zeros(a.K)
    t_last, tok_last, t_ckpt = time.time(), st["tokens"], time.time()
    bench_t0 = bench_tok0 = None
    tp_t0, tp_tok0 = time.time(), st["tokens"]       # throughput, measured after the first 10 steps
    while st["tokens"] < T:
        frac = st["tokens"] / T
        lr = lr_at(frac, a.lr)
        for g in opt.param_groups:
            g["lr"] = lr
        eps = eps_at(frac, a.eps_start, a.eps_final, a.eps_anneal)
        for _ in range(a.accum):
            b = next(loader)
            x, y = b["x"].to(dev, non_blocking=True), b["y"].to(dev, non_blocking=True)
            ver = b["verify"].to(dev)
            with autocast(dev):
                z, winner = choose_z(fmodel, model, x, y, a.z, a.method, a.K, eps, a.balance)
                nll = per_example_nll(fmodel(x, z), y)
            (nll.mean() / a.accum).backward()
            st["tokens"] += b["ntok"]
            with torch.no_grad():
                agg["s"] += float(nll[~ver].sum()); agg["ns"] += int((~ver).sum())
                agg["v"] += float(nll[ver].sum()); agg["nv"] += int(ver.sum())
                if winner is not None and a.z == "fix":
                    wc = torch.bincount(winner, minlength=a.K).cpu().float()
                    agg["win"] += wc
                    win_total += wc
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        st["step"] += 1
        if st["step"] == 10:
            tp_t0, tp_tok0 = time.time(), st["tokens"]
        if a.bench_steps:
            if st["step"] == 10:
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                bench_t0, bench_tok0 = time.time(), st["tokens"]
            if st["step"] == 10 + a.bench_steps:
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                tps = (st["tokens"] - bench_tok0) / (time.time() - bench_t0)
                write_json(os.path.join(a.out, "bench.json"), {"tok_per_s": tps})
                log(f"bench {a.method}/{a.z}: {tps:.0f} tok/s")
                mark_done(a.out, {"tok_per_s": tps})
                return
        if st["step"] % a.log_every == 0:
            now = time.time()
            tps = (st["tokens"] - tok_last) / (now - t_last)
            t_last, tok_last = now, st["tokens"]
            ls = agg["s"] / max(agg["ns"], 1)
            lv = agg["v"] / max(agg["nv"], 1)
            wins = (agg["win"] / agg["win"].sum().clamp(min=1)).tolist()
            w.writerow([st["step"], st["tokens"], f"{lr:.2e}", f"{eps:.3f}", f"{ls:.4f}", f"{lv:.4f}",
                        f"{tps:.0f}"] + [f"{x:.3f}" for x in wins])
            logf.flush()
            eta = (T - st["tokens"]) / max(tps, 1) / 3600
            log(f"step {st['step']} tok {st['tokens'] / 1e9:.3f}/{T / 1e9:.2f}B loss_solve {ls:.4f} "
                f"loss_verify {lv:.4f} {tps / 1e3:.0f}k tok/s eta {eta:.1f}h" +
                (f" wins {[round(x, 2) for x in wins]}" if a.z == "fix" and a.method == "wta" else ""))
            agg = {"s": 0.0, "ns": 0, "v": 0.0, "nv": 0, "win": torch.zeros(a.K)}
        if a.early_frac and not st["early_saved"] and st["tokens"] >= a.early_frac * T:
            save_model(os.path.join(a.out, "early.pt"), model, dict(meta, tokens_seen=st["tokens"]))
            st["early_saved"] = True
        if time.time() - t_ckpt > a.ckpt_minutes * 60:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": st}, last + ".tmp")
            os.replace(last + ".tmp", last)
            t_ckpt = time.time()
    tok_per_s = (st["tokens"] - tp_tok0) / max(time.time() - tp_t0, 1e-6)
    save_model(os.path.join(a.out, "final.pt"), model, dict(meta, tokens_seen=st["tokens"]))
    accs, gps = quick_eval(model, a.z, max(a.op_lo, a.op_max - 4), a.op_max, a.quick_eval_n)
    log(f"quick eval (greedy, val op {max(a.op_lo, a.op_max - 4)}..{a.op_max}): {accs}  gen {gps:.0f}/s")
    res = {"tokens": st["tokens"], "steps": st["step"], "val_acc_per_mode": accs,
           "val_acc": sum(accs) / len(accs), "gen_per_s": gps,
           "win_share": (win_total / win_total.sum().clamp(min=1)).tolist() if a.z == "fix" else None,
           "tok_per_s": tok_per_s}
    if os.path.exists(last):
        os.remove(last)
    mark_done(a.out, res)
    log("done", a.out)


if __name__ == "__main__":
    cli_main(main)
