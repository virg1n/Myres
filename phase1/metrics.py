"""Load phase-1 evaluation outputs into the array format of phase0.report / pilot.metrics."""
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase0.report import arrays, phi_within  # noqa: E402
from pilot.metrics import auroc  # noqa: E402


def read_gz(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(l) for l in f]


def load_eval(d, rows, setname, with_verify=True):
    """(A, bins, missing_yes_no, verify_ok) for one evaluated model on one set, or None if the
    evaluation is incomplete (no manifest, fewer modes than evaluated, set missing).
    verify_ok=False -> Yes/No metrics must be reported as unavailable (p_err is NaN)."""
    mf = os.path.join(d, "eval.json")
    if not os.path.exists(mf):
        return None
    man = json.load(open(mf))
    if setname not in man["sets"]:
        return None
    single = os.path.join(d, f"solve_single_{setname}.jsonl.gz")
    if os.path.exists(single):
        neu = read_gz(single)
        sol = [[[(r["answers"][i + 4 * s], r["correct"][i + 4 * s], r["texts"][i + 4 * s]) for r in neu]
                for s in range(2)] for i in range(4)]
        vf = os.path.join(d, "verify_single.json")
        ok = with_verify and os.path.exists(vf)
        if ok:
            v = json.load(open(vf))
            p, miss = [[v["p_no"][i]] * 4 for i in range(4)], v["missing"]
        else:
            p, miss = np.full((4, 4, len(rows)), np.nan).tolist(), 0
        return arrays(rows, sol, p) + (miss, ok)
    M = man["modes"]
    if not all(os.path.exists(os.path.join(d, f"solve_{k}_{setname}.jsonl.gz")) for k in range(M)):
        return None
    so = [read_gz(os.path.join(d, f"solve_{k}_{setname}.jsonl.gz")) for k in range(M)]
    sol = [[[(r["answers"][s], r["correct"][s], r["texts"][s]) for r in so[k]] for s in range(2)] for k in range(M)]
    miss = 0
    ok = with_verify and all(os.path.exists(os.path.join(d, f"verify_{j}.json")) for j in range(M))
    if ok:
        vs = [json.load(open(os.path.join(d, f"verify_{j}.json"))) for j in range(M)]
        p = [[vs[j]["p_no"][i] for j in range(M)] for i in range(M)]
        miss = sum(v["missing"] for v in vs)
    else:
        p = np.full((M, M, len(rows)), np.nan).tolist()
    return arrays(rows, sol, p) + (miss, ok)


def graded_disagreement(A, idx=None):
    """Score of solution (i, sample 0) = share of the other modes whose answer differs (missing
    answer counts as disagreement). Returns (scores [M,N], labels [M,N])."""
    ans, err = A["answer"][:, 0], A["has_err"][:, 0]
    if idx is not None:
        ans, err = ans[:, idx], err[:, idx]
    M = ans.shape[0]
    sc = np.zeros(ans.shape)
    for i in range(M):
        others = [j for j in range(M) if j != i]
        sc[i] = np.mean([(ans[j] != ans[i]) | (ans[i] < 0) for j in others], axis=0)
    return sc, err


def graded_auroc(A, idx=None):
    sc, err = graded_disagreement(A, idx)
    return auroc(sc.ravel(), err.ravel())


def graded_at(A, thr):
    """TPR / FPR of the rule 'flag if at least thr of the other modes disagree'."""
    sc, err = graded_disagreement(A)
    f = sc >= thr - 1e-9
    return float(f[err].mean()) if err.any() else float("nan"), float(f[~err].mean()) if (~err).any() else float("nan")


def val_summary(A):
    wrong = ~A["correct"][:, 0]
    return {"pass1": float(A["correct"][:, 0].mean()), "phi": phi_within(wrong, A["op"], np.arange(wrong.shape[1])),
            "modes_differ": float(np.mean([len(set(A["answer"][:, 0, n])) > 1 for n in range(wrong.shape[1])]))}
