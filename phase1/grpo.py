"""Stage 2 by RL: GRPO from a phase-1 adapter, with or without modes (phase1/grun.py runs the chains).

Step t takes P problems of the training pool (pool.jsonl without the problems that every M_0 pool attempt of --m0
solved; consecutive slices of a fresh permutation per epoch, seeded by --seed: the same problems in the same order for
every chain of a seed) and samples G rollouts of each from the current policy (on-policy, HF generate inside the
trainer: T = 1, no top-p / top-k, at most --max_tokens new tokens, the sampling seed of the step fixed by (seed, t);
FiLM codes per batch row, separate adapters one mode per call). Reward r = 1 if the answer is right and in \\boxed{}
(selftrain.admitted, the rule of the self-training bank), else 0. Dr. GRPO advantage, no division by the std and no
length normalization:
    A_i = r_i - mean of the G rewards of the problem          (equal rewards: A = 0, no gradient)
One optimizer step per batch of rollouts, so the ratio to the sampling policy is 1 and PPO clipping is the identity;
no KL to the start:
    L = - sum_i A_i sum_t log pi(o_it | q, o_i<t) / (P G max_tokens)       (one constant normalizer for every step)
Modes (K = 4 FiLM codes or separate LoRA adapters): G / K rollouts per mode. The group is still all G rollouts of the
problem with the same A_i as without modes; rollout i trains only through its own mode: its FiLM code (gamma of code
k on every norm, as --variant fixed in phase1/train.py; the LoRA and the FiLM MLP are shared) or its adapter.
Why this analog of "a solution trains the mode that found it with weight 1/|W|": in RFT, 1/|W| is the weight
1/occupied slots of rftbal (each solved problem weighs 1), so wta1 and rftbal differed only in the routing. Here the
per-problem accounting is the group baseline; it is the same G-rollout group in both arms, and only the routing
differs. In expectation a baseline does not change the gradient: each mode maximizes its own pass@1, the objective of
plain GRPO, so nothing but the routing keeps the modes apart (their collapse is measured by the report). Per-mode
groups of G / K would make the groups smaller than in plain GRPO (more groups with equal rewards, a noisier
baseline); an explicit 1/|W| adds an objective plain GRPO does not have. That one is the ablation --adv cover:
    A_i <- A_i prod_{j != k(i)} (1 - p_j),   p_j the share of right rollouts of mode j in the group,
an unbiased estimate of the gradient of the coverage 1 - prod_k (1 - p_k) of one attempt per mode (pass@k training
across modes), where --adv mean follows sum_k p_k.

Steps: --steps S, or the steps pinned in --steps_file; the first chain to start computes them from the calibration runs
(--calib_from: calib.json of `--calib n`, minutes per step) so that the slowest calibrated chain fits --train_hours,
and every chain and seed trains the same S. Checkpoints ckpt{j}/adapter.pt at S j / --evals (j = 1..evals) in the
format of phase1.train ({"adapter", "cfg"}), for phase1.selftrain gen. Resume: last.pt (adapter, Adam, step, RNG
states) with every checkpoint and every --ckpt_minutes; steps.jsonl and rollouts.jsonl are cut back to the resumed
step, whose rollouts are drawn again with the same seed. --max_hours is a safety limit: a run stopped by it writes its
last adapter as the final checkpoint, result.json says so, and last.pt stays (removing DONE resumes)."""
import argparse
import functools
import gzip
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase0 import answers, prompts  # noqa: E402
from phase1 import film as FM  # noqa: E402
from phase1.keval import check_manifest  # noqa: E402
from phase1.selftrain import admitted, attempts  # noqa: E402
from phase1.train import collate, load_jsonl, log, micro, per_example_nll  # noqa: E402

CALIB_MARGIN = 1.1          # S fits --train_hours with this margin over the calibrated minutes per step


# ----------------------------------------------------------------------------- the math
def reward(text, answer):
    """1 for a right answer in \\boxed{} (the bank rule of self-training), else 0."""
    k = answers.key(answers.extract(text))
    return float(admitted(text, k != "" and k == answers.key(answer)))


def group_adv(r, modes=None, K=0, kind="mean"):
    """Advantages of the G rollouts of one problem: r - mean(r); cover: times prod_{j != own mode} (1 - p_j)."""
    r = np.asarray(r, dtype=float)
    A = r - r.mean()
    if kind == "cover":
        m = np.asarray(modes)
        miss = np.array([1 - r[m == k].mean() for k in range(K)])
        A = A * np.array([np.prod(np.delete(miss, k)) for k in range(K)])[m]
    return A


def pg_coef(adv, max_tokens):
    """Weight of the summed NLL of every rollout of a step, A_i / (P G max_tokens): one constant normalizer."""
    return np.asarray(adv, dtype=float) / (len(adv) * max_tokens)


@functools.lru_cache(maxsize=4)
def _perm(n, seed, epoch):
    order = list(range(n))
    random.Random(seed * 1_000_003 + epoch).shuffle(order)
    return order


def step_problems(n, seed, P, step):
    """Pool indices of the P problems of a step: consecutive slices of a fresh permutation per epoch."""
    return [_perm(n, seed, i // n)[i % n] for i in range(step * P, (step + 1) * P)]


def gen_seed(seed, step):
    return 1_000_003 * (seed + 1) + step


def ckpt_steps(S, n):
    return [round(S * j / n) for j in range(1, n + 1)]


def plan_steps(calib, train_hours, n, steps_max):
    """Steps that the slowest calibrated chain does in train_hours, a multiple of n (the checkpoints)."""
    mps = max(c["min_per_step"] for c in calib)
    S = min(steps_max, int(train_hours * 60 / (CALIB_MARGIN * mps)))
    return max(n, S // n * n)


def pinned_steps(a):
    """S of every chain and seed: read from --steps_file, or computed from --calib_from and written there once."""
    if not a.steps_file:
        raise SystemExit("give --steps or --steps_file (with --calib_from)")
    if not os.path.exists(a.steps_file):
        files = [f for f in a.calib_from.split(",") if f]
        missing = [f for f in files if not os.path.exists(f)]
        if not files or missing:
            raise SystemExit(f"no steps: {a.steps_file} does not exist and calibration is missing {missing or '(none given)'}")
        cal = [json.load(open(f)) for f in files]
        S = plan_steps(cal, a.train_hours, a.evals, a.steps_max)
        tmp = f"{a.steps_file}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump({"steps": S, "train_hours": a.train_hours, "steps_max": a.steps_max, "evals": a.evals,
                       "min_per_step": {fn: c["min_per_step"] for fn, c in zip(files, cal)}}, f, indent=1)
        try:                   # an atomic create: the chains start together, the first one pins S (the same for all)
            os.link(tmp, a.steps_file)
        except FileExistsError:
            pass
        except OSError:        # no hard links on this file system
            if not os.path.exists(a.steps_file):
                os.replace(tmp, a.steps_file)
        if os.path.exists(tmp):
            os.remove(tmp)
    return json.load(open(a.steps_file))["steps"]


def train_pool(data, m0, drop_easy):
    """(problems, dropped): pool.jsonl without the problems that every M_0 pool attempt of the given generation
    directories solved (too easy: a group of equal rewards); the same for every chain of a seed."""
    pool = load_jsonl(os.path.join(data, "pool.jsonl"))
    if not (drop_easy and m0):
        return pool, 0
    cnt = {}
    for g in m0:
        for i, lst in attempts(g, "pool").items():
            c = cnt.setdefault(i, [0, 0])
            c[0] += sum(admitted(t, ok) for _, t, ok, _ in lst)
            c[1] += len(lst)
    keep = [p for p in pool if not (cnt.get(p["id"], [0, 0])[1] and cnt[p["id"]][0] == cnt[p["id"]][1])]
    return keep, len(pool) - len(keep)


# ----------------------------------------------------------------------------- rollouts and the gradient
def stop_ids(tok):
    ids = [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")] + [tok.eos_token_id]
    return sorted({i for i in ids if isinstance(i, int) and i != tok.unk_token_id})


def gen_config(a, pad, stops):
    """Sampling of the rollouts. Set as the model's own generation config too: generate() otherwise fills values equal
    to the library defaults (temperature 1, top_p 1) from generation_config.json of the base (Qwen3: 0.6, 0.95)."""
    from transformers import GenerationConfig
    kw = dict(do_sample=True, temperature=a.temp, top_p=a.top_p, top_k=a.top_k) if a.temp > 0 else dict(do_sample=False)
    gc = GenerationConfig(max_new_tokens=a.max_tokens, eos_token_id=stops, pad_token_id=pad, use_cache=True, **kw)
    if a.cache == "static":
        gc.cache_implementation = "static"
    return gc


@torch.no_grad()
def rollout(model, film, ctx, rows, gc, pad, dev, batch, per_mode, tokens=0):
    """rows: dicts p (prompt ids), mode (None without modes) -> [(completion ids, truncated)] per row: the sampled
    tokens up to and including the first stop token (truncated: none within max_new_tokens). Left-padded batches of
    at most `batch` rows and (tokens > 0) at most `tokens` cache positions (rows x (longest prompt + max_new_tokens):
    the KV cache), rows sorted by prompt length; FiLM codes per row (ctx.gamma); separate adapters one mode per call
    (per_mode): ctx.modes of size 1 broadcasts, where a per-row gather would copy every adapter for every row at
    every token."""
    model.eval()
    stops = set(gc.eos_token_id)
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["mode"] if per_mode else 0, len(rows[i]["p"]), i))
    calls = []
    for i in order:
        if calls and len(calls[-1]) < batch and (not per_mode or rows[calls[-1][0]]["mode"] == rows[i]["mode"]) and \
                (not tokens or (len(calls[-1]) + 1) * (len(rows[i]["p"]) + gc.max_new_tokens) <= tokens):
            calls[-1].append(i)
        else:
            calls.append([i])
    out = [None] * len(rows)
    dt = torch.device(dev).type
    for ids in calls:
        L = max(len(rows[i]["p"]) for i in ids)
        x = torch.full((len(ids), L), pad, dtype=torch.long)
        att = torch.zeros((len(ids), L), dtype=torch.long)
        for r, i in enumerate(ids):
            p = rows[i]["p"]
            x[r, L - len(p):] = torch.tensor(p)
            att[r, L - len(p):] = 1
        modes = [rows[i]["mode"] for i in ids]
        ctx.gamma, ctx.modes = None, None
        with torch.autocast(dt, dtype=torch.bfloat16, enabled=dt == "cuda"):
            if modes[0] is not None:      # gamma under autocast, as per_example_nll computes it in training
                if per_mode:
                    ctx.modes = torch.tensor([modes[0]], device=dev)
                else:
                    ctx.gamma = film.gamma(torch.tensor(modes, device=dev))
            seq = model.generate(input_ids=x.to(dev), attention_mask=att.to(dev), generation_config=gc)
        for r, t in zip(ids, seq[:, L:].tolist()):
            cut = next((j for j, v in enumerate(t) if v in stops), None)
            out[r] = (t[:cut + 1], False) if cut is not None else (t, True)
    ctx.gamma, ctx.modes = None, None
    return out


def backward_pg(model, film, ctx, rows, coef, pad, dev, mb_tokens):
    """Accumulates the gradient of sum_i coef_i * NLL_i, NLL_i the negative log-likelihood of completion t of row i
    after its prompt p, summed over the completion tokens, through the mode of the row only (one mode per micro-batch:
    a FiLM code or an adapter of size 1 broadcasts). Returns NLL_i and the completion tokens of every row."""
    nll, ntok = [0.0] * len(rows), [len(r["t"]) for r in rows]
    by = {}
    for i, r in enumerate(rows):
        by.setdefault(r["mode"], []).append(i)
    model.train()
    dt = torch.device(dev).type
    for m, idx in by.items():
        sub = [rows[i] for i in idx]
        md = None if m is None else torch.tensor([m], device=dev)
        for ids in micro(sub, mb_tokens):
            x, att, lab = collate(sub, ids, pad, dev)
            with torch.autocast(dt, dtype=torch.bfloat16, enabled=dt == "cuda"):
                mean = per_example_nll(model, film, ctx, x, att, lab, md)
            tot = mean.float() * torch.tensor([ntok[idx[j]] for j in ids], device=dev, dtype=torch.float32)
            (tot * torch.tensor([coef[idx[j]] for j in ids], device=dev, dtype=torch.float32)).sum().backward()
            for j, v in zip(ids, tot.detach().cpu().tolist()):
                nll[idx[j]] = v
    ctx.gamma, ctx.modes = None, None
    return nll, ntok


# ----------------------------------------------------------------------------- files
def save_atomic(obj, path):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def trim_log(path, step):
    """Keep the records of steps before `step` (a resumed run repeats the later ones); drops a torn last line."""
    if not os.path.exists(path):
        return
    keep = []
    with open(path) as f:
        for line in f:
            try:
                if json.loads(line)["step"] < step:
                    keep.append(line if line.endswith("\n") else line + "\n")
            except (ValueError, KeyError):
                pass
    with open(path + ".tmp", "w") as f:
        f.writelines(keep)
    os.replace(path + ".tmp", path)


def append_jsonl(path, recs):
    with open(path, "a") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_gz(path, rows):
    with gzip.open(path + ".tmp", "wt") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(path + ".tmp", path)


def file_id(path):
    return {"path": os.path.abspath(path), "bytes": os.path.getsize(path), "mtime": int(os.path.getmtime(path))}


# ----------------------------------------------------------------------------- main
def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--init", required=True, help="start adapter: FiLM codes, separate adapters or no modes")
    ap.add_argument("--data", required=True, help="dir with pool.jsonl")
    ap.add_argument("--m0", default="", help="comma-separated M_0 generation dirs with pool attempts (--drop_easy)")
    ap.add_argument("--drop_easy", type=int, default=1, help="1: no problems solved by every M_0 pool attempt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0, help="problem order and rollout seeds (the same for every chain)")
    ap.add_argument("--G", type=int, default=16, help="rollouts per problem (a multiple of the modes)")
    ap.add_argument("--P", type=int, default=8, help="problems per optimizer step")
    ap.add_argument("--steps", type=int, default=0, help="optimizer steps; 0: --steps_file / --calib_from")
    ap.add_argument("--steps_file", default="")
    ap.add_argument("--calib_from", default="", help="comma-separated calib.json: S fits the slowest into --train_hours")
    ap.add_argument("--train_hours", type=float, default=9.0)
    ap.add_argument("--steps_max", type=int, default=1000)
    ap.add_argument("--evals", type=int, default=3, help="checkpoints ckpt1..ckptN at S j / N")
    ap.add_argument("--calib", type=int, default=0, help="only run this many steps and write calib.json")
    ap.add_argument("--adv", choices=["mean", "cover"], default="mean")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--film_lr_ratio", type=float, default=5.0, help="FiLM LR / LoRA LR (phase 1: 1e-3 / 2e-4)")
    ap.add_argument("--warmup", type=int, default=10, help="linear warmup steps, then a constant LR")
    ap.add_argument("--clip", type=float, default=1.0, help="gradient norm clipping")
    ap.add_argument("--temp", type=float, default=1.0, help="rollout temperature: 1 (the loss differentiates log pi)")
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=1200)
    ap.add_argument("--gen_batch", type=int, default=128, help="rows per generate call")
    ap.add_argument("--gen_tokens", type=int, default=128 * 1800,
                    help="KV cache positions per generate call: rows x (longest prompt + max_tokens), ~26 GB at 230K")
    ap.add_argument("--cache", choices=["dynamic", "static"], default="dynamic", help="KV cache of generate")
    ap.add_argument("--mb_tokens", type=int, default=6144, help="padded tokens per training micro-batch")
    ap.add_argument("--max_hours", type=float, default=48.0, help="safety limit of this training")
    ap.add_argument("--ckpt_minutes", type=float, default=20.0, help="last.pt this often (and at every checkpoint)")
    ap.add_argument("--keep_texts", type=int, default=50, help="texts of the rollouts of every n-th step (0: none)")
    ap.add_argument("--stop_after", type=int, default=0, help=argparse.SUPPRESS)      # tests: as if killed after n steps
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    os.makedirs(a.out, exist_ok=True)
    if os.path.exists(os.path.join(a.out, "DONE")):
        return "done"
    from transformers import AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    init = torch.load(a.init, map_location="cpu", weights_only=False)
    c0 = init["cfg"]
    if c0.get("prefix_modes") or c0.get("film_z"):
        raise SystemExit("GRPO supports FiLM codes, separate adapters or no modes")
    film_modes, lora_modes = c0.get("film_modes", 0), c0.get("lora_modes", 0)
    K = film_modes or lora_modes
    if K and a.G % K:
        raise SystemExit(f"--G {a.G} is not a multiple of the {K} modes")
    if a.adv == "cover" and not K:
        raise SystemExit("--adv cover needs modes")
    if (a.temp, a.top_p, a.top_k) != (1.0, 1.0, 0):
        raise SystemExit("rollouts must sample pi itself (--temp 1, --top_p 1, --top_k 0): the loss differentiates "
                         "log pi, so another sampling distribution makes the gradient off-policy")
    tok = AutoTokenizer.from_pretrained(a.base)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    stops = stop_ids(tok)
    m0 = [d for d in a.m0.split(",") if d]
    pool, dropped = train_pool(a.data, m0, a.drop_easy)
    if not pool:
        raise SystemExit("the training pool is empty")
    S = a.calib or a.steps or pinned_steps(a)
    if S < a.evals and not a.calib:
        raise SystemExit(f"{S} steps for {a.evals} checkpoints")
    ckpts = [] if a.calib else ckpt_steps(S, a.evals)
    pool_hash = hashlib.sha1(",".join(p["id"] for p in pool).encode()).hexdigest()
    check_manifest(a.out, {"init": file_id(a.init), "base": os.path.abspath(a.base), "data": os.path.abspath(a.data),
                           "m0": [os.path.abspath(d) for d in m0], "drop_easy": a.drop_easy,
                           "pool": [len(pool), pool_hash], "seed": a.seed, "G": a.G, "P": a.P, "steps": S,
                           "evals": a.evals, "calib": a.calib, "adv": a.adv, "lr": a.lr, "film_lr_ratio": a.film_lr_ratio,
                           "warmup": a.warmup, "clip": a.clip, "temp": a.temp, "top_p": a.top_p, "top_k": a.top_k,
                           "max_tokens": a.max_tokens, "gen_batch": a.gen_batch, "gen_tokens": a.gen_tokens,
                           "cache": a.cache})
    r, alpha = c0["lora_r"], c0["lora_alpha"]
    model, film, ctx = FM.build(a.base, film_modes, r, alpha, dev, lora_modes=lora_modes)
    FM.load_adapter(model, film, init["adapter"])
    del init                          # a phase-1 adapter may carry its Adam moments
    lora_ps, film_ps = FM.trainable(model, film)
    groups = [{"params": lora_ps, "lr": a.lr}] + ([{"params": film_ps, "lr": a.lr * a.film_lr_ratio}] if film_ps else [])
    opt = torch.optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.99))
    cfg = {"variant": "grpo", "film_modes": film_modes, "lora_r": r, "lora_alpha": alpha, "lora_modes": lora_modes,
           "prefix_modes": 0, "film_z": 0}
    st = {"step": 0, "elapsed": 0.0, "rollouts": 0, "gen_tokens": 0, "train_tokens": 0, "step_s": []}
    last = os.path.join(a.out, "last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=False)
        FM.load_adapter(model, film, ck["adapter"])
        opt.load_state_dict(ck["opt"])
        st = ck["state"]
        torch.set_rng_state(ck["rng"]["torch"])
        if dev == "cuda" and ck["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
        random.setstate(ck["rng"]["py"])
        np.random.set_state(ck["rng"]["np"])
        log(f"resumed at step {st['step']}")
        del ck
    fs, fr = os.path.join(a.out, "steps.jsonl"), os.path.join(a.out, "rollouts.jsonl")
    for f in (fs, fr):
        trim_log(f, st["step"])
    gc = gen_config(a, pad, stops)
    model.generation_config = gc
    n_tr = sum(p.numel() for p in lora_ps + film_ps)
    kind = "FiLM codes" if film_modes else "separate adapters" if lora_modes else "no modes"
    log(f"GRPO from {a.init}: {kind}" + (f" (K = {K})" if K else "") + f", trainable {n_tr / 1e6:.1f}M; pool "
        f"{len(pool)} problems ({dropped} easy dropped); {S} steps x {a.P} problems x {a.G} rollouts, adv {a.adv}, "
        f"LR {a.lr:g}, checkpoints at {ckpts}, device {dev}")

    def save_last():
        save_atomic({"adapter": FM.adapter_state(model, film), "opt": opt.state_dict(), "state": st, "cfg": cfg,
                     "rng": {"torch": torch.get_rng_state(), "py": random.getstate(), "np": np.random.get_state(),
                             "cuda": torch.cuda.get_rng_state_all() if dev == "cuda" else None}}, last)

    def save_ckpt(j):
        d = os.path.join(a.out, f"ckpt{j}")
        os.makedirs(d, exist_ok=True)
        save_atomic({"adapter": FM.adapter_state(model, film), "cfg": cfg, "step": st["step"]},
                    os.path.join(d, "adapter.pt"))

    prompt_ids = {}

    def pids(p):
        if p["id"] not in prompt_ids:
            prompt_ids[p["id"]] = tok.encode(prompts.solve_prompt(tok, p["question"], "neutral"), add_special_tokens=False)
        return prompt_ids[p["id"]]

    t_run, el0, n_run, t_save = time.time(), st["elapsed"], 0, time.time()
    stopped = None
    while st["step"] < S:
        if el0 + time.time() - t_run > a.max_hours * 3600:
            stopped = "time"
            log(f"safety limit of {a.max_hours} h reached at step {st['step']} of {S}")
            break
        t = st["step"]
        t0 = time.time()
        probs = [pool[i] for i in step_problems(len(pool), a.seed, a.P, t)]
        rows = [{"id": p["id"], "answer": p["answer"], "p": pids(p), "mode": g // (a.G // K) if K else None}
                for p in probs for g in range(a.G)]
        torch.manual_seed(gen_seed(a.seed, t))
        outs = rollout(model, film, ctx, rows, gc, pad, dev, a.gen_batch, per_mode=bool(lora_modes), tokens=a.gen_tokens)
        if dev == "cuda":
            torch.cuda.empty_cache()
        gen_s = time.time() - t0
        for row, (ids, cut) in zip(rows, outs):
            text = tok.decode(ids, skip_special_tokens=True)
            row.update(t=ids, trunc=cut, text=text, r=reward(text, row["answer"]), boxed=answers.last_boxed(text) is not None,
                       key=answers.key(answers.extract(text)))
        adv = np.concatenate([group_adv([x["r"] for x in rows[i:i + a.G]], [x["mode"] for x in rows[i:i + a.G]], K, a.adv)
                              for i in range(0, len(rows), a.G)])
        mult = min(1.0, (t + 1) / max(a.warmup, 1))
        for g, base_lr in zip(opt.param_groups, [a.lr, a.lr * a.film_lr_ratio]):
            g["lr"] = base_lr * mult
        coef = pg_coef(adv, a.max_tokens)
        live = [i for i in range(len(rows)) if coef[i] != 0]
        t1 = time.time()
        nll, ntok = backward_pg(model, film, ctx, [rows[i] for i in live], [coef[i] for i in live], pad, dev,
                                a.mb_tokens) if live else ([], [])
        gn = float(torch.nn.utils.clip_grad_norm_(lora_ps + film_ps, a.clip)) if live else 0.0
        opt.step()
        opt.zero_grad(set_to_none=True)
        if dev == "cuda":
            torch.cuda.empty_cache()
        train_s = time.time() - t1
        rs = np.array([x["r"] for x in rows]).reshape(a.P, a.G)
        rec = {"step": t, "lr": a.lr * mult, "reward": float(rs.mean()), "mixed": float(np.mean(rs.std(1) > 0)),
               "all0": float(np.mean(rs.max(1) == 0)), "all1": float(np.mean(rs.min(1) == 1)),
               "len": float(np.mean([len(x["t"]) for x in rows])), "trunc": float(np.mean([x["trunc"] for x in rows])),
               "noboxed": float(np.mean([not x["boxed"] for x in rows])), "trained": len(live),
               "nll": sum(nll) / sum(ntok) if ntok else None, "grad_norm": gn, "gen_s": gen_s, "train_s": train_s,
               "gen_tokens": sum(len(x["t"]) for x in rows)}
        if K:
            rec["reward_mode"] = [float(np.mean([x["r"] for x in rows if x["mode"] == k])) for k in range(K)]
        append_jsonl(fs, [rec])
        append_jsonl(fr, [{"step": t, "id": x["id"], "mode": x["mode"], "r": x["r"], "len": len(x["t"]),
                           "trunc": x["trunc"], "boxed": x["boxed"], "key": x["key"]} for x in rows])
        if a.keep_texts and not a.calib and (t % a.keep_texts == 0 or t == S - 1):
            os.makedirs(os.path.join(a.out, "texts"), exist_ok=True)
            write_gz(os.path.join(a.out, "texts", f"step{t:05d}.jsonl.gz"),
                     [{"id": x["id"], "mode": x["mode"], "r": x["r"], "trunc": x["trunc"], "text": x["text"]} for x in rows])
        st["step"] += 1
        st["elapsed"] = el0 + time.time() - t_run
        st["rollouts"] += len(rows)
        st["gen_tokens"] += rec["gen_tokens"]
        st["train_tokens"] += sum(ntok)
        st["step_s"].append(time.time() - t0)
        n_run += 1
        mps = float(np.mean(st["step_s"][-20:])) / 60
        log(f"step {st['step']}/{S}: reward {rec['reward']:.3f}" + (f" {[round(x, 2) for x in rec['reward_mode']]}" if K else "")
            + f", mixed groups {rec['mixed']:.2f}, len {rec['len']:.0f}, cut {rec['trunc']:.2f}, no boxed "
            f"{rec['noboxed']:.2f} | gen {gen_s:.0f} s ({len(rows) / max(gen_s, 1e-9) * 60:.0f} rollouts/min, "
            f"{rec['gen_tokens'] / max(gen_s, 1e-9):.0f} tok/s), train {train_s:.0f} s | {mps:.2f} min/step, "
            f"{st['elapsed'] / 3600:.2f} h, ETA {(S - st['step']) * mps / 60:.1f} h")
        if st["step"] in ckpts:
            save_ckpt(ckpts.index(st["step"]) + 1)
            save_last()
            t_save = time.time()
        elif not a.calib and time.time() - t_save > a.ckpt_minutes * 60:
            save_last()
            t_save = time.time()
        if a.stop_after and n_run >= a.stop_after and st["step"] < S:
            save_last()
            log(f"stopped after {n_run} steps (--stop_after)")
            return "stopped"
    if a.calib:
        dts = st["step_s"][1:] or st["step_s"]       # the first step also warms up CUDA and the allocator
        json.dump({"min_per_step": float(np.mean(dts)) / 60, "steps": st["step"], "step_s": st["step_s"],
                   "rollouts_per_step": a.P * a.G, "init": os.path.abspath(a.init), "max_tokens": a.max_tokens,
                   "gen_tokens": st["gen_tokens"]}, open(os.path.join(a.out, "calib.json"), "w"), indent=1)
    else:
        if stopped and st["step"] and not os.path.exists(os.path.join(a.out, f"ckpt{a.evals}", "adapter.pt")):
            save_ckpt(a.evals)            # the last adapter as the final checkpoint; result.json marks it
        json.dump({"steps": S, "steps_done": st["step"], "complete": st["step"] == S, "stopped": stopped,
                   "hours": st["elapsed"] / 3600, "rollouts": st["rollouts"], "gen_tokens": st["gen_tokens"],
                   "train_tokens": st["train_tokens"], "min_per_step": float(np.mean(st["step_s"] or [0])) / 60,
                   "trainable_M": n_tr / 1e6, "pool": len(pool), "dropped_easy": dropped, "modes": K, "kind": kind,
                   "adv": a.adv, "seed": a.seed, "P": a.P, "G": a.G, "ckpts": ckpts},
                  open(os.path.join(a.out, "result.json"), "w"), indent=1)
    if os.path.exists(last) and not stopped:       # stopped by --max_hours: last.pt stays (remove DONE to resume)
        os.remove(last)
    open(os.path.join(a.out, "DONE"), "w").close()
    log("done", a.out)
    return "done"


if __name__ == "__main__":
    main()
