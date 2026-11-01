"""Does the Yes/No verifier read the solution, or only the problem? And does a mode that solved the
problem itself catch another mode's error better? Uses saved outputs only (no GPU).

    python3 -m phase1.verify_analysis --run runs/phase1 --phase0 runs/phase0

1. within-problem AUROC: only pairs (wrong, correct) of the SAME problem scored by the same verifier
   -> difficulty cannot help; ~0.5 means the verifier does not distinguish solutions.
   difficulty-only AUROC: every solution gets the verifier's mean score over its problem
   -> how much of the overall AUROC difficulty alone explains.
2. AUROC without "empty" errors (no extracted answer / truncated).
3. the hypothesis cell: solver i, other verifier j != i, on problems where j's own solution is
   correct vs where j is wrong too; compared with i verifying itself on the very same solutions.
   Fresh eyes should help in the first group: AUROC(j) - AUROC(i) > 0.
Intervals: 95% bootstrap over problems.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase0.report import load_methods  # noqa: E402
from phase1.metrics import load_eval  # noqa: E402
from pilot.metrics import auroc  # noqa: E402

BOOT = 300
VARIANTS = ["sft1", "sftmulti", "filmmean", "wta1", "wtamulti", "wtabal", "s2_best", "s2_ctrl"]


def within_problem(A, idx):
    """Pairwise AUROC restricted to (wrong, correct) pairs of the same problem and verifier."""
    p, err = A["p_err"], A["has_err"][:, 0]
    M = p.shape[0]
    num = den = 0.0
    for n in idx:
        e = err[:, n]
        if e.all() or not e.any():
            continue
        for j in range(M):
            s = p[:, j, n]
            w, r = s[e], s[~e]
            diff = w[:, None] - r[None, :]
            num += (diff > 0).sum() + 0.5 * (diff == 0).sum()
            den += diff.size
    return num / den if den else float("nan")


def pooled(A, idx, fn):
    """AUROC over all (solver i, verifier j, problem n) items selected by fn(i, j, n) -> bool mask over n."""
    p, err = A["p_err"], A["has_err"][:, 0]
    M = p.shape[0]
    sc, lab = [], []
    for i in range(M):
        for j in range(M):
            m = fn(i, j)
            sel = idx[m[idx]]
            sc.append(p[i, j, sel])
            lab.append(err[i, sel])
    sc, lab = np.concatenate(sc), np.concatenate(lab)
    return auroc(sc, lab)


def difficulty_only(A, idx):
    p, err = A["p_err"], A["has_err"][:, 0]
    M = p.shape[0]
    prob = p.mean(0)                              # [J, N]: verifier j's mean score on problem n
    sc = np.concatenate([prob[j, idx] for i in range(M) for j in range(M)])
    lab = np.concatenate([err[i, idx] for i in range(M) for j in range(M)])
    return auroc(sc, lab)


def cell_delta(A, idx, j_correct):
    """AUROC(other verifier j) - AUROC(self verifier i) on solver i's solutions for problems where
    j's own sample-0 answer is correct (j_correct=True) or wrong (False). Pooled over i != j."""
    p, err = A["p_err"], A["has_err"][:, 0]
    M = p.shape[0]
    so, ss, lab = [], [], []
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            m = (~err[j] if j_correct else err[j])[idx]
            sel = idx[m]
            so.append(p[i, j, sel])
            ss.append(p[i, i, sel])
            lab.append(err[i, sel])
    so, ss, lab = np.concatenate(so), np.concatenate(ss), np.concatenate(lab)
    return auroc(so, lab) - auroc(ss, lab), auroc(so, lab), auroc(ss, lab), int(lab.sum()), int((~lab).sum())


def boot(fn, N, seed=0):
    rng = np.random.default_rng(seed)
    pt = fn(np.arange(N))
    vals = [fn(rng.integers(0, N, N)) for _ in range(BOOT)]
    vals = [v for v in vals if np.isfinite(v)]
    if not vals or not np.isfinite(pt):
        return pt, float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(pt), float(lo), float(hi)


def f(x):
    return "—" if not np.isfinite(x) else f"{x:.3f}"


def c(t):
    return "—" if not np.isfinite(t[0]) else f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def analyse(name, A, multi):
    N = A["p_err"].shape[2]
    full = np.arange(N)
    err = A["has_err"][:, 0]
    empty = A["answer"][:, 0] < 0                 # wrong because no answer was extracted
    out = {"name": name, "errors": int(err.sum()), "empty_errors": int((err & empty).sum())}
    out["auroc_all"] = pooled(A, full, lambda i, j: np.ones(N, bool))
    out["auroc_diff_only"] = difficulty_only(A, full)
    out["within"] = boot(lambda ix: within_problem(A, ix), N)
    A2 = dict(A, has_err=A["has_err"].copy())
    keep = ~(err & empty)                         # drop "empty" wrong solutions
    out["auroc_no_empty"] = pooled(A2, full, lambda i, j: keep[i])
    if multi:
        out["delta_j_correct"] = boot(lambda ix: cell_delta(A, ix, True)[0], N)
        out["delta_j_wrong"] = boot(lambda ix: cell_delta(A, ix, False)[0], N)
        _, ao, as_, npos, nneg = cell_delta(A, full, True)
        out["j_correct_detail"] = {"auroc_other": ao, "auroc_self": as_, "errors": npos, "correct": nneg}
        _, ao, as_, npos, nneg = cell_delta(A, full, False)
        out["j_wrong_detail"] = {"auroc_other": ao, "auroc_self": as_, "errors": npos, "correct": nneg}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/phase1")
    ap.add_argument("--phase0", default="runs/phase0")
    a = ap.parse_args()
    test = [json.loads(l) for l in open(os.path.join(a.run, "data", "test.jsonl"))]
    methods = []
    if os.path.isdir(a.phase0):
        _, m0 = load_methods(a.phase0)
        for k in ("temperature", "strategies"):
            if k in m0:
                methods.append((f"p0_{k}", m0[k][0], k == "strategies"))
    for v in VARIANTS:
        r = load_eval(os.path.join(a.run, "eval", v), test, "test")
        if r is not None and r[3]:
            single = os.path.exists(os.path.join(a.run, "eval", v, "solve_single_test.jsonl.gz"))
            methods.append((v, r[0], not single))
    res = [analyse(n, A, multi) for n, A, multi in methods]
    L = ["# Анализ проверяющего Yes/No\n",
         "## 1–2. Читает ли проверяющий решение\n",
         "«Внутри задачи» — AUROC только по парам (неверное, верное) решения **одной и той же** задачи: сложность "
         "задачи здесь не помогает, 0.5 = проверяющий не различает решения. «Только сложность» — каждому решению "
         "дана средняя оценка проверяющего по его задаче.\n",
         "| метод | ошибок | из них без ответа | AUROC всё | AUROC «только сложность» | AUROC внутри задачи | "
         "AUROC без ошибок «без ответа» |", "|---|---|---|---|---|---|---|"]
    for r in res:
        L.append(f"| {r['name']} | {r['errors']} | {r['empty_errors']} | {f(r['auroc_all'])} | {f(r['auroc_diff_only'])} | "
                 f"{c(r['within'])} | {f(r['auroc_no_empty'])} |")
    L += ["\n## 3. Помогает ли «свежий взгляд»\n",
          "Решения режима i. Сравниваем проверяющего j ≠ i с самим i на **тех же** решениях. Группа «j решил сам»: "
          "задачи, где у j свой ответ верный — здесь гипотеза предсказывает Δ > 0. Группа «j тоже ошибся» — "
          "контроль.\n",
          "| метод | j решил сам: Δ AUROC (j − i) | AUROC j / i | ошибок / верных | j тоже ошибся: Δ AUROC (j − i) | "
          "AUROC j / i | ошибок / верных |", "|---|---|---|---|---|---|---|"]
    for r in res:
        if "delta_j_correct" not in r:
            continue
        d1, d2 = r["j_correct_detail"], r["j_wrong_detail"]
        L.append(f"| {r['name']} | {c(r['delta_j_correct'])} | {f(d1['auroc_other'])} / {f(d1['auroc_self'])} | "
                 f"{d1['errors']} / {d1['correct']} | {c(r['delta_j_wrong'])} | {f(d2['auroc_other'])} / "
                 f"{f(d2['auroc_self'])} | {d2['errors']} / {d2['correct']} |")
    L += ["\n## Как читать\n",
          "- AUROC «только сложность» близок к «всё», а «внутри задачи» близок к 0.5 → проверяющий оценивает "
          "сложность задачи, а не решение; тогда отсутствие перекрёстного эффекта ничего не говорит об идее.\n"
          "- «Внутри задачи» заметно выше 0.5 → решение он действительно читает.\n"
          "- Раздел 3: Δ > 0 с интервалом выше нуля в группе «j решил сам» → свежий взгляд помогает именно там, где "
          "должен. Δ ≈ 0 в обеих группах → не помогает."]
    out = os.path.join(a.run, "verify_analysis.md")
    with open(out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("\nwritten:", out)


if __name__ == "__main__":
    main()
