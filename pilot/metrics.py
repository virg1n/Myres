"""Metrics from raw evaluation arrays (saved per model and test set in arrays_<set>.npz).

Arrays (M modes, S samples per mode, N problems):
  correct[M,S,N]  answer[M,S,N]  has_err[M,S,N]  first_err[M,S,N] (0 = none)  err_pid[M,S,N] (-1)
  backward[M,S,N] sol_hash[M,N] op[N]
  p_err[I,J,N]   verifier J's P(error) for solution (I, sample 0)
  loc[I,J,N]     verifier J's predicted first wrong step
  cmp_flag[I,J,N], cmp_loc[I,J,N]  "solve yourself and compare" protocol (J==I uses sample 1)
Per-problem vectors are returned so that report.py can bootstrap paired differences.
"""
import numpy as np

FPR_TARGET = 0.05


def rankdata(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return ranks


def auroc(scores, labels):
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = rankdata(np.concatenate([neg, pos]))
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def thresholds(p_err, err0):
    """Threshold per cell (solver i, verifier j) and fold (problem index parity): 95% quantile of
    the cell's scores on correct solutions of the *other* fold (cross-fitting), so every cell has
    FPR ~ 5% out of sample and TPRs are comparable across cells."""
    I, J, N = p_err.shape
    fold = np.arange(N) % 2
    thr = np.full((I, J, 2), 0.5)
    for i in range(I):
        for j in range(J):
            for f in (0, 1):
                s = p_err[i, j][(fold != f) & ~err0[i]]
                if len(s):
                    thr[i, j, f] = np.quantile(s, 1 - FPR_TARGET)
    return thr, fold


def flags(p_err, err0):
    thr, fold = thresholds(p_err, err0)
    return p_err > thr[:, :, fold]


def majority(answers, weights=None):
    """answers [K,N] int (-1 = none) -> chosen answer per problem (ties: earliest)."""
    K, N = answers.shape
    out = np.full(N, -1)
    for n in range(N):
        score = {}
        for k in range(K):
            a = answers[k, n]
            if a < 0:
                continue
            score[a] = score.get(a, 0.0) + (1.0 if weights is None else weights[k, n])
        if score:
            best = max(score.values())
            for k in range(K):
                if answers[k, n] in score and score[answers[k, n]] == best:
                    out[n] = answers[k, n]
                    break
    return out


def per_problem(A):
    """Per-problem vectors (each length N) for everything report needs."""
    correct, answer, has_err = A["correct"], A["answer"], A["has_err"]
    M, S, N = correct.shape
    truth = A["truth"]
    err0 = has_err[:, 0, :]                       # [M,N] label of the verified solution
    first0 = A["first_err"][:, 0, :]
    v = {}
    v["pass1"] = correct[:, 0, :].mean(0)
    order = [(k, s) for s in range(S) for k in range(M)]   # one per mode first
    for kk in (2, 4, 8, 16):
        if kk <= len(order):
            sel = order[:kk]
            v[f"pass@{kk}"] = np.any([correct[k, s] for k, s in sel], axis=0).astype(float)
    v["maj4"] = (majority(answer[:, 0, :]) == truth).astype(float)
    v["maj16"] = (majority(answer.reshape(M * S, N)) == truth).astype(float)
    p_err, loc = A["p_err"], A["loc"]
    flag = flags(p_err, err0)                                               # [I,J,N]
    for name, F, L in (("ver", flag, loc), ("cmp", A["cmp_flag"], A["cmp_loc"])):
        for kind, cells in (("diag", [(i, i) for i in range(M)]),
                            ("off", [(i, j) for i in range(M) for j in range(M) if i != j])):
            tp = np.zeros(N); pos = np.zeros(N); fp = np.zeros(N); neg = np.zeros(N)
            lc = np.zeros(N); ln = np.zeros(N)
            for i, j in cells:
                e = err0[i]
                tp += F[i, j] & e; pos += e
                fp += F[i, j] & ~e; neg += ~e
                lm = e & (first0[i] <= 64) & (first0[i] > 0)
                lc += (L[i, j] == first0[i]) & lm; ln += lm
            v[f"{name}_{kind}_tp"], v[f"{name}_{kind}_pos"] = tp, pos
            v[f"{name}_{kind}_fp"], v[f"{name}_{kind}_neg"] = fp, neg
            v[f"{name}_{kind}_lc"], v[f"{name}_{kind}_ln"] = lc, ln
    # end-to-end procedures with 4 solutions + verification
    off_mean = np.stack([np.mean([p_err[i, j] for j in range(M) if j != i], 0) for i in range(M)])
    self_s = np.stack([p_err[i, i] for i in range(M)])
    v["vvote_cross"] = (majority(answer[:, 0, :], 1 - off_mean) == truth).astype(float)
    v["vvote_self"] = (majority(answer[:, 0, :], 1 - self_s) == truth).astype(float)
    acc, fixed, broken = np.zeros(N), np.zeros(N), np.zeros(N)
    pairs = [(i, j) for i in range(M) for j in range(M) if i != j]
    for i, j in pairs:
        new = np.where(flag[i, j], answer[j, 0], answer[i, 0]) == truth
        acc += new; fixed += new & ~correct[i, 0]; broken += ~new & correct[i, 0]
    v["switch_acc"], v["switch_fixed"], v["switch_broken"] = acc / len(pairs), fixed / len(pairs), broken / len(pairs)
    return v


def ratio(v, a, b, idx=None):
    na, nb = (v[a], v[b]) if idx is None else (v[a][idx], v[b][idx])
    return float(na.sum() / nb.sum()) if nb.sum() > 0 else float("nan")


def summarize(A):
    """Scalar metrics for one model and test set."""
    v = per_problem(A)
    M = A["correct"].shape[0]
    out = {k: float(v[k].mean()) for k in ("pass1", "maj4", "maj16", "vvote_cross", "vvote_self",
                                         "switch_acc", "switch_fixed", "switch_broken") if k in v}
    for kk in (2, 4, 8, 16):
        if f"pass@{kk}" in v:
            out[f"pass@{kk}"] = float(v[f"pass@{kk}"].mean())
    for name in ("ver", "cmp"):
        for kind in ("diag", "off"):
            out[f"{name}_tpr_{kind}"] = ratio(v, f"{name}_{kind}_tp", f"{name}_{kind}_pos")
            out[f"{name}_fpr_{kind}"] = ratio(v, f"{name}_{kind}_fp", f"{name}_{kind}_neg")
            out[f"{name}_loc_{kind}"] = ratio(v, f"{name}_{kind}_lc", f"{name}_{kind}_ln")
    err0 = A["has_err"][:, 0, :]
    au = np.full((M, M), np.nan)
    tpr = np.full((M, M), np.nan)
    fl = flags(A["p_err"], err0)
    for i in range(M):
        for j in range(M):
            au[i, j] = auroc(A["p_err"][i, j], err0[i])
            tpr[i, j] = fl[i, j][err0[i]].mean() if err0[i].any() else np.nan
    out["auroc_matrix"] = au.tolist()
    out["tpr_matrix"] = tpr.tolist()
    with np.errstate(all="ignore"):
        d, o = np.diag(au), au[~np.eye(M, dtype=bool)]
        out["auroc_diag"] = float(np.nanmean(d)) if np.isfinite(d).any() else float("nan")
        out["auroc_off"] = float(np.nanmean(o)) if np.isfinite(o).any() else float("nan")
    out["n_errors_mode0"] = int(err0[0].sum())
    out["n_errors_total"] = int(err0.sum())
    out["acc_per_mode"] = A["correct"][:, 0, :].mean(1).tolist()
    out["backward_share_per_mode"] = A["backward"][:, 0, :].mean(1).tolist()
    out.update(diagnostics(A))
    return out


def diagnostics(A):
    correct, err_pid, op = A["correct"][:, 0, :], A["err_pid"][:, 0, :], A["op"]
    M, N = correct.shape
    wrong = ~correct
    # error correlation within difficulty bins (phi), averaged over mode pairs and bins
    phis = []
    for o in np.unique(op):
        m = op == o
        if m.sum() < 20:
            continue
        for i in range(M):
            for j in range(i + 1, M):
                a, b = wrong[i, m].astype(float), wrong[j, m].astype(float)
                if a.std() > 0 and b.std() > 0:
                    phis.append(np.corrcoef(a, b)[0, 1])
    same_pid, both = 0, 0
    for i in range(M):
        for j in range(i + 1, M):
            m = wrong[i] & wrong[j] & (err_pid[i] >= 0) & (err_pid[j] >= 0)
            both += m.sum()
            same_pid += (err_pid[i][m] == err_pid[j][m]).sum()
    ans = A["answer"][:, 0, :]
    differ_ans = np.mean([len(set(ans[:, n])) > 1 for n in range(N)])
    differ_sol = np.mean([len(set(A["sol_hash"][:, n])) > 1 for n in range(N)])
    return {"err_phi_within_op": float(np.mean(phis)) if phis else float("nan"),
            "same_first_error_param": float(same_pid / both) if both else float("nan"),
            "modes_differ_answer": float(differ_ans), "modes_differ_solution": float(differ_sol)}
