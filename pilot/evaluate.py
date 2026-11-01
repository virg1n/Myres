"""Evaluate one method: solutions per mode, "who verifies whom" matrix, end-to-end procedures.

Modes (M=4):
  fix codes      -> code k
  continuous z   -> 4 fixed draws z_k (plus a separate pass@k curve with fresh z per sample)
  baseline       -> 4 independent samples at the tuned temperature (one verifier)
  --noise fix    -> baseline weights + fixed noise vector k (B-fix)
  --noise cont   -> baseline weights + fresh noise per generation chunk (B-cont)
Temperature (and noise scale for B) are tuned on a val split by majority-of-4 accuracy.
"""
import argparse
import os
import time

import numpy as np
import torch

from . import igsm
from .common import cli_main, device, is_done, load_model, log, mark_done, write_json
from .metrics import majority, summarize
from .model import score_positions

M, S = 4, 4


class ModelAgent:
    deterministic = True      # greedy decoding gives the same solution for the same mode

    def __init__(self, model, z_mode, seed, gen_bs):
        self.model, self.zm, self.gen_bs = model, z_mode, gen_bs
        g = torch.Generator().manual_seed(seed)
        self.zbank = torch.randn(M, model.cfg.zdim, generator=g)

    def z(self, mode, n):
        if self.zm == "none":
            return None
        if self.zm == "fix":
            return self.model.code_z(torch.full((n,), mode, dtype=torch.long,
                                                device=self.model.emb.weight.device)).detach()
        return self.zbank[mode].unsqueeze(0).expand(n, -1).contiguous()

    def solve(self, prompts, mode, temp, seed):
        return self.model.generate(prompts, z=self.z(mode, len(prompts)), temperature=temp,
                                   eos_id=igsm.EOS, batch_size=self.gen_bs, seed=seed)

    def solve_fresh(self, prompts, temp, seed):
        g = torch.Generator().manual_seed(seed)
        z = torch.randn(len(prompts), self.model.cfg.zdim, generator=g)
        return self.model.generate(prompts, z=z, temperature=temp, eos_id=igsm.EOS,
                                   batch_size=self.gen_bs, seed=seed)

    def score(self, seqs, positions, mode):
        return score_positions(self.model, seqs, positions, z=self.z(mode, len(seqs)))


class NoiseAgent:
    """Gaussian noise on all transformer Linear weights: W += sigma * std(W) * eps."""

    def __init__(self, model, sigma, fresh, seed, gen_bs):
        self.model, self.sigma, self.fresh, self.seed, self.gen_bs = model, sigma, fresh, seed, gen_bs
        self.deterministic = not fresh
        self.names = [n for n, p in model.named_parameters() if n.startswith("blocks.") and p.dim() == 2]
        params = dict(model.named_parameters())
        self.orig = {n: params[n].detach().clone() for n in self.names}
        self.calls = 0

    @torch.no_grad()
    def _apply(self, nseed):
        params = dict(self.model.named_parameters())
        g = torch.Generator(device=self.model.emb.weight.device).manual_seed(nseed)
        for n in self.names:
            w0 = self.orig[n]
            params[n].copy_(w0 + self.sigma * w0.std() * torch.randn(w0.shape, device=w0.device, generator=g))

    @torch.no_grad()
    def _restore(self):
        params = dict(self.model.named_parameters())
        for n in self.names:
            params[n].copy_(self.orig[n])

    def _chunked(self, items, mode, fn):
        self.calls += 1
        if not self.fresh:
            self._apply(self.seed * 1000 + mode)
            out = fn(items)
            self._restore()
            return out
        out = []
        for c in range(0, len(items), self.gen_bs):
            self._apply(self.seed * 1000003 + self.calls * 7919 + c + 17 * mode)
            out += fn(items[c:c + self.gen_bs])
        self._restore()
        return out

    def solve(self, prompts, mode, temp, seed):
        return self._chunked(prompts, mode, lambda p: self.model.generate(
            p, temperature=temp, eos_id=igsm.EOS, batch_size=self.gen_bs, seed=seed))

    def score(self, seqs, positions, mode):
        pairs = list(zip(seqs, positions))
        return self._chunked(pairs, mode, lambda pp: score_positions(
            self.model, [s for s, _ in pp], [q for _, q in pp]))


def check_all(probs, outs):
    return [igsm.check_solution(p, o) for p, o in zip(probs, outs)]


def tune(agent, probs, temps, seed):
    prompts = [igsm.solve_prompt(p) for p in probs]
    truth = np.array([p.answer for p in probs])
    best = None
    rows = []
    for t in temps:
        ans = np.full((M, len(probs)), -1)
        for k in range(M):
            outs = agent.solve(prompts, k, t, seed + k)
            ans[k] = [c["answer"] if c["answer"] is not None else -1 for c in check_all(probs, outs)]
        maj = float((majority(ans) == truth).mean())
        p1 = float((ans == truth[None]).mean())
        rows.append({"temp": t, "maj4": maj, "pass1": p1})
        if best is None or maj > best["maj4"] + 1e-9:
            best = rows[-1]
    return best, rows


def run_set(agent, probs, temp, seed, max_len, fresh_curve):
    N = len(probs)
    prompts = [igsm.solve_prompt(p) for p in probs]
    truth = np.array([p.answer for p in probs])
    A = {k: np.zeros((M, S, N), dtype=dt) for k, dt in
         (("correct", bool), ("answer", np.int64), ("has_err", bool), ("first_err", np.int64),
          ("err_pid", np.int64), ("backward", bool))}
    A["sol_hash"] = np.zeros((M, N), dtype=np.int64)
    gens, checks = {}, {}
    for k in range(M):
        for s in range(S):
            if s > 0 and temp == 0 and getattr(agent, "deterministic", False):
                outs, ch = gens[k, 0], checks[k, 0]          # greedy: identical to sample 0
            else:
                outs = agent.solve(prompts, k, temp, seed + 100 * k + s)
                ch = check_all(probs, outs)
            gens[k, s], checks[k, s] = outs, ch
            for n, c in enumerate(ch):
                A["correct"][k, s, n] = c["correct"]
                A["answer"][k, s, n] = c["answer"] if c["answer"] is not None else -1
                A["has_err"][k, s, n] = c["has_error"]
                A["first_err"][k, s, n] = c["first_err"] or 0
                A["err_pid"][k, s, n] = c["err_pid"] if c["err_pid"] is not None else -1
                A["backward"][k, s, n] = c["style"] == "backward"
        A["sol_hash"][k] = [hash(tuple(o)) for o in gens[k, 0]]
    # protocol 1: read the solution and verify (trained verifier head)
    A["p_err"] = np.zeros((M, M, N))
    A["loc"] = np.zeros((M, M, N), dtype=np.int64)
    for i in range(M):
        seqs, pos = [], []
        for p, o in zip(probs, gens[i, 0]):
            pre = igsm.verify_prefix(p, o)[:max_len - 2]
            seqs.append(pre + [igsm.VER, igsm.ERR])
            pos.append([len(pre), len(pre) + 1])
        for j in range(M):
            lps = agent.score(seqs, pos, j)
            for n, lp in enumerate(lps):
                a, b = float(lp[0, igsm.OK]), float(lp[0, igsm.ERR])
                A["p_err"][i, j, n] = 1.0 / (1.0 + np.exp(a - b))
                A["loc"][i, j, n] = int(lp[1, igsm.STEP_IDS].argmax()) + 1
    # protocol 2: solve independently, compare answers / step values
    A["cmp_flag"] = np.zeros((M, M, N), dtype=bool)
    A["cmp_loc"] = np.zeros((M, M, N), dtype=np.int64)
    for i in range(M):
        for j in range(M):
            other = checks[j, 0] if j != i else checks[i, 1]
            for n in range(N):
                ci, cj = checks[i, 0][n], other[n]
                flag = ci["answer"] != cj["answer"]
                loc = 0
                for step, pid, v in ci["steps"]:
                    if pid in cj["values"] and cj["values"][pid] != v:
                        loc = step
                        break
                if flag and loc == 0:
                    loc = ci["n_steps"] + 1
                A["cmp_flag"][i, j, n], A["cmp_loc"][i, j, n] = flag, loc
    A["op"] = np.array([p.op for p in probs])
    A["truth"] = truth
    if fresh_curve is not None:
        fc = np.zeros((16, N), dtype=bool)
        for s in range(16):
            outs = fresh_curve(prompts, temp, seed + 5000 + s)
            fc[s] = [c["correct"] for c in check_all(probs, outs)]
        A["fresh_correct"] = fc
    return A, gens


def dump_examples(path, probs, gens, A, n=5):
    with open(path, "w") as f:
        for q in range(min(n, len(probs))):
            p = probs[q]
            f.write(f"=== problem {q} (op={p.op}, answer={p.answer})\n{igsm.decode(p.tokens())}\n")
            for k in range(M):
                f.write(f"--- mode {k}: correct={A['correct'][k, 0, q]} first_err={A['first_err'][k, 0, q]} "
                        f"p_err by verifiers={np.round(A['p_err'][k, :, q], 3).tolist()}\n")
                f.write(igsm.decode(gens[k, 0][q]) + "\n")
            f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--noise", choices=["", "fix", "cont"], default="")
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--n_val", type=int, default=400)
    ap.add_argument("--gen_bs", type=int, default=512)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.01, 0.02, 0.04, 0.08])
    ap.add_argument("--seed", type=int, default=2024)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if is_done(a.out):
        return
    t0 = time.time()
    model, meta = load_model(a.ckpt, device())
    model.eval()
    zm, op_max = meta["z"], meta["op_max"]
    lo = max(meta["op_lo"], op_max - 4)
    L = model.cfg.max_len
    val = igsm.fixed_set(11, "val", a.n_val, lo, op_max, L)
    sets = {"id": igsm.fixed_set(22, "test", a.n_test, lo, op_max, L),
            "ood": igsm.fixed_set(33, "test", a.n_test, op_max + 1, op_max + 4, L)}
    info = {"ckpt": a.ckpt, "meta": meta, "noise": a.noise}
    if a.noise:
        base = ModelAgent(model, "none", a.seed, a.gen_bs)
        greedy = check_all(val, base.solve([igsm.solve_prompt(p) for p in val], 0, 0.0, a.seed))
        g_acc = float(np.mean([c["correct"] for c in greedy]))
        cands = []
        for sg in a.sigmas:
            agent = NoiseAgent(model, sg, a.noise == "cont", a.seed, a.gen_bs)
            best, rows = tune(agent, val, [0.0, 0.6], a.seed)
            for r in rows:
                cands.append(dict(r, sigma=sg))
        ok = [c for c in cands if c["pass1"] >= g_acc - 0.03] or [max(cands, key=lambda c: c["pass1"])]
        best = max(ok, key=lambda c: (c["maj4"], -c["sigma"]))
        info.update(greedy_val_acc=g_acc, tuning=cands, sigma=best["sigma"], temp=best["temp"])
        agent = NoiseAgent(model, best["sigma"], a.noise == "cont", a.seed, a.gen_bs)
        temp = best["temp"]
    else:
        agent = ModelAgent(model, zm, a.seed, a.gen_bs)
        temps = [0.3, 0.6, 0.9] if zm == "none" else [0.0, 0.3, 0.6, 0.9]
        best, rows = tune(agent, val, temps, a.seed)
        info.update(tuning=rows, temp=best["temp"])
        temp = best["temp"]
    log(f"eval {a.out}: tuned {info.get('sigma', '')} temp={temp}")
    metrics = {}
    for name, probs in sets.items():
        fresh = agent.solve_fresh if (not a.noise and zm == "cont") else None
        A, gens = run_set(agent, probs, temp, a.seed + 7, L, fresh)
        np.savez_compressed(os.path.join(a.out, f"arrays_{name}.npz"), **A)
        metrics[name] = summarize(A)
        if "fresh_correct" in A:
            fc = A["fresh_correct"]
            metrics[name]["fresh_pass@k"] = {k: float(fc[:k].any(0).mean()) for k in (1, 2, 4, 8, 16)}
        dump_examples(os.path.join(a.out, f"examples_{name}.txt"), probs, gens, A)
        m = metrics[name]
        log(f"[{name}] pass1 {m['pass1']:.3f} maj4 {m['maj4']:.3f} pass@16 {m.get('pass@16', 0):.3f} | "
            f"TPR@5%FPR diag {m['ver_tpr_diag']:.3f} off {m['ver_tpr_off']:.3f} | "
            f"AUROC diag {m['auroc_diag']:.3f} off {m['auroc_off']:.3f}")
    info["metrics"] = metrics
    info["eval_minutes"] = (time.time() - t0) / 60
    write_json(os.path.join(a.out, "metrics.json"), info)
    mark_done(a.out)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    cli_main(main)
