"""Training-example builders and the packed-batch collate function.

An example is one packed sequence: the causal stream (prompt + solution) plus
any number of *window copies*. Three kinds of windows are produced:

* clean    -- window over a correct solution: no errors, identity fix targets.
* masked   -- window over a correct solution in which the tokens that would be
              erased after a fix are replaced by <mask> (plus optional slack
              masks that must be filled with <null>): teaches regeneration.
* error    -- window over the model's *own* rollout containing its first
              mistake j: error labels, fix target set at j, and a dependency
              row j (filled by the oracle DAG or by the confidence-drop
              measurement in the trainer).
"""

import random

import numpy as np
import torch

from .igsm.checker import ERR, OK, descendants

IGNORE = -100


def window_start(state_sol, W, mask_id):
    """Window over the solution part of the current text: last W tokens,
    extended to the left so that it always contains every <mask>."""
    ws = max(0, len(state_sol) - W)
    for i, t in enumerate(state_sol):
        if t == mask_id:
            return min(ws, i)
    return ws


def runs_of(positions):
    """Sorted positions -> list of (start, end_inclusive) contiguous runs."""
    out = []
    for p in sorted(set(positions)):
        if out and p == out[-1][1] + 1:
            out[-1][1] = p
        else:
            out.append([p, p])
    return [tuple(r) for r in out]


def erase(sol, j, D, mask_id, slack=0, slack_min_len=3):
    """Apply an erasure to a solution token list after a fix at j.
    D: positions (> j) to erase. The run touching the end is truncated; every
    other run becomes <mask> tokens (+ `slack` extra masks for long runs).
    Returns (new_sol, list of (orig_start, orig_end, mask_start, n_masks))."""
    D = sorted(k for k in set(D) if j < k < len(sol))
    runs = runs_of(D)
    if runs and runs[-1][1] == len(sol) - 1:
        sol = sol[: runs[-1][0]]
        runs = runs[:-1]
    out, info, prev = [], [], 0
    for a, b in runs:
        out += sol[prev:a]
        n = b - a + 1 + (slack if b - a + 1 >= slack_min_len else 0)
        info.append((a, b, len(out), n))
        out += [mask_id] * n
        prev = b + 1
    out += sol[prev:]
    return out, info


def _window(start_abs, copy_ids, err, fix, fix_w, dep_row=-1, dep_tgt=None, confdrop=None):
    return {"start": start_abs, "ids": copy_ids, "err": err, "fix": fix, "fix_w": fix_w,
            "dep_row": dep_row, "dep_tgt": dep_tgt, "confdrop": confdrop}


def plain_example(prob):
    ids = prob.prompt_ids + prob.solution_ids
    P = len(prob.prompt_ids)
    lm = [IGNORE] * len(ids)
    for i in range(P - 1, len(ids) - 1):
        lm[i] = ids[i + 1]
    return {"ids": ids, "lm_tgt": lm, "windows": [], "kind": "plain"}


def clean_window(prob, W, rng, id_w, parents=None, dep_rows=0):
    sol = prob.solution_ids
    P = len(prob.prompt_ids)
    t = rng.randint(1, len(sol))
    ws = max(0, t - W)
    copy = sol[ws:t]
    n = len(copy)
    win = _window(P + ws, copy, [0] * n, [[x] for x in copy], [id_w] * n)
    if dep_rows and parents is not None and n >= 2:
        # dense oracle dependency supervision (only used with dep_target=oracle)
        j = rng.randint(ws, t - 2)
        d = set(descendants(parents[:t], j))
        win["dep_row"] = j - ws
        win["dep_tgt"] = [(-1 if k <= j else int(k in d)) for k in range(ws, t)]
    return win


def masked_window(prob, parents, lines, W, rng, tok, cfg):
    sol = prob.solution_ids
    P = len(prob.prompt_ids)
    if len(sol) < 4:
        return None
    t = rng.randint(min(len(sol), 6), len(sol))
    ws = max(0, t - W)
    j = rng.randint(ws, t - 2)
    pat = rng.choices(["dag", "span", "line", "random"], weights=cfg["mask_patterns"])[0]
    if pat == "dag":
        D = [k for k in descendants(parents[:t], j) if k < t]
    elif pat == "span":
        a = rng.randint(j + 1, t - 1)
        D = list(range(a, min(t - 1, a + rng.randint(0, 11)) + 1))
    elif pat == "line":
        cand = sorted({lines[k] for k in range(j + 1, t)})
        ln = rng.choice(cand)
        D = [k for k in range(j + 1, t) if lines[k] == ln]
    else:
        m = rng.randint(1, max(1, min(8, t - 1 - j)))
        D = rng.sample(range(j + 1, t), m)
    D = sorted(set(D))
    runs = runs_of(D)
    if runs and runs[-1][1] == t - 1:  # tail run is truncated at inference, not masked
        t = runs[-1][0]
        D = [k for k in D if k < t]
    if not D or t - j < 2:
        return None
    state, info = erase(sol[:t], j, D, tok.mask, cfg["slack"], cfg["slack_min_len"])
    # targets in state coordinates
    tgt = [[x] for x in state]
    is_mask = [False] * len(state)
    for (a, b, ms, n) in info:
        orig = sol[a: b + 1]
        for i in range(n):
            tgt[ms + i] = [orig[i]] if i < len(orig) else [tok.null]
            is_mask[ms + i] = True
    ws2 = window_start(state, W, tok.mask)
    copy = state[ws2:]
    err = [-1 if is_mask[ws2 + i] else 0 for i in range(len(copy))]
    fix = tgt[ws2:]
    fix_w = [1.0 if is_mask[ws2 + i] else cfg["identity_w"] for i in range(len(copy))]
    return _window(P + ws2, copy, err, fix, fix_w)


def correct_example(prob, analysis, rng, tok, cfg, n_clean, n_masked):
    ex = plain_example(prob)
    ex["kind"] = "correct"
    W = cfg["window"]
    dense = cfg.get("dep_target") == "oracle"
    for _ in range(n_clean):
        ex["windows"].append(clean_window(prob, W, rng, cfg["identity_w"],
                                          analysis.parents if dense else None, dep_rows=int(dense)))
    for _ in range(n_masked):
        w = masked_window(prob, analysis.parents, analysis.lines, W, rng, tok, cfg)
        if w is None:
            w = clean_window(prob, W, rng, cfg["identity_w"])
        ex["windows"].append(w)
    return ex


def error_example(entry, rng, tok, cfg):
    """entry: dict(prompt, sol, status, valid, parents, j) from the rollout buffer."""
    W = cfg["window"]
    prompt, sol, status, valid, parents, j = (entry[k] for k in ("prompt", "sol", "status", "valid", "parents", "j"))
    P = len(prompt)
    t = rng.randint(j + 1, min(len(sol), j + W))
    t_neg = rng.randint(1, j) if j >= 1 and rng.random() < cfg.get("neg_window_prob", 0.5) else None
    ids = prompt + sol[:t]
    lm = [IGNORE] * len(ids)
    if cfg.get("lm_on_self", False):
        for i in range(P - 1, P + j - 1):
            lm[i] = ids[i + 1]
    ex = {"ids": ids, "lm_tgt": lm, "windows": [], "kind": "error"}

    ws = max(0, t - W)
    copy = sol[ws:t]
    err, fix, fix_w = [], [], []
    for k in range(ws, t):
        s = status[k]
        if s == ERR:
            err.append(1)
            fix.append(list(valid[k]) if valid[k] else [])
            fix_w.append(1.0 if valid[k] else 0.0)
        elif s == OK:
            err.append(0)
            fix.append([sol[k]])
            fix_w.append(cfg["identity_w"])
        else:
            err.append(-1)
            fix.append([])
            fix_w.append(0.0)
    win = _window(P + ws, copy, err, fix, fix_w)
    if t - 1 > j and valid[j]:
        win["dep_row"] = j - ws
        if cfg.get("dep_target") == "oracle":
            d = set(descendants(parents[:t], j))
            win["dep_tgt"] = [(-1 if k <= j else int(k in d)) for k in range(ws, t)]
        else:
            win["confdrop"] = {"fix_tok": rng.choice(valid[j])}
            win["dep_tgt"] = [-1] * len(copy)
    ex["windows"].append(win)

    if t_neg is not None:
        ws_n = max(0, t_neg - W)
        c = sol[ws_n:t_neg]
        ex["windows"].append(_window(P + ws_n, c, [0] * len(c), [[x] for x in c],
                                     [cfg["identity_w"]] * len(c)))
    return ex


# ---------------------------------------------------------------------------
# collate
# ---------------------------------------------------------------------------

def collate(examples, pad_id, max_set=64):
    B = len(examples)
    lens = [len(e["ids"]) + sum(len(w["ids"]) for w in e["windows"]) for e in examples]
    L = max(lens)
    ids = np.full((B, L), pad_id, dtype=np.int64)
    pos = np.zeros((B, L), dtype=np.int64)
    is_copy = np.zeros((B, L), dtype=bool)
    wstart = np.zeros((B, L), dtype=np.int64)
    wid = np.full((B, L), -1, dtype=np.int64)
    valid = np.zeros((B, L), dtype=bool)
    lm_tgt = np.full((B, L), IGNORE, dtype=np.int64)
    err_tgt = np.full((B, L), -1, dtype=np.int64)
    fix_pos, fix_sets, fix_w = [], [], []
    dep_q, dep_k, dep_t = [], [], []
    confdrop = []  # (row index in dep arrays, b, copy offset, copy len, j, fix_tok)
    for b, e in enumerate(examples):
        n = len(e["ids"])
        ids[b, :n] = e["ids"]
        pos[b, :n] = np.arange(n)
        valid[b, :n] = True
        lm_tgt[b, :n] = e["lm_tgt"]
        off = n
        for wi, w in enumerate(e["windows"]):
            m = len(w["ids"])
            if m == 0:
                continue
            ids[b, off:off + m] = w["ids"]
            pos[b, off:off + m] = np.arange(w["start"], w["start"] + m)
            is_copy[b, off:off + m] = True
            wstart[b, off:off + m] = w["start"]
            wid[b, off:off + m] = wi
            valid[b, off:off + m] = True
            err_tgt[b, off:off + m] = w["err"]
            base = b * L + off
            for i, (fs, fw) in enumerate(zip(w["fix"], w["fix_w"])):
                if fw > 0 and fs:
                    fix_pos.append(base + i)
                    fix_sets.append(fs)
                    fix_w.append(fw)
            if w.get("dep_row", -1) >= 0 and w.get("dep_tgt") is not None:
                r = len(dep_q)
                dep_q.append(base + w["dep_row"])
                dep_k.append((base, m))
                dep_t.append(w["dep_tgt"])
                if w.get("confdrop") is not None:
                    confdrop.append((r, b, off, m, w["dep_row"], w["confdrop"]["fix_tok"]))
            off += m

    T = torch.from_numpy
    batch = {"ids": T(ids), "pos": T(pos), "is_copy": T(is_copy), "wstart": T(wstart), "wid": T(wid),
             "valid": T(valid), "lm_tgt": T(lm_tgt), "err_tgt": T(err_tgt),
             "kinds": [e.get("kind", "") for e in examples], "confdrop": confdrop}
    S = min(max_set, max([len(x) for x in fix_sets] + [1]))
    fs = np.full((len(fix_sets), S), -1, dtype=np.int64)
    single = [i for i, x in enumerate(fix_sets) if len(x) == 1]
    if single:
        fs[single, 0] = [fix_sets[i][0] for i in single]
    for i, x in enumerate(fix_sets):
        if len(x) > 1:
            x = x[:S]
            fs[i, :len(x)] = x
    batch["fix_pos"] = torch.tensor(fix_pos, dtype=torch.long)
    batch["fix_sets"] = T(fs)
    batch["fix_w"] = torch.tensor(fix_w, dtype=torch.float)
    K = max([m for _, m in dep_k] + [1])
    dk = np.full((len(dep_k), K), -1, dtype=np.int64)
    dt = np.full((len(dep_k), K), -1, dtype=np.int64)
    for i, ((base, m), t) in enumerate(zip(dep_k, dep_t)):
        dk[i, :m] = np.arange(base, base + m)
        dt[i, :len(t)] = t
    batch["dep_q"] = torch.tensor(dep_q, dtype=torch.long)
    batch["dep_k"] = T(dk)
    batch["dep_t"] = T(dt)
    return batch


def tokens_in_batch(batch):
    return int(batch["valid"].sum())
