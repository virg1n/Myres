"""Report of the private-share sweep (phase1/prun.py) -> report_priv.md, summary_priv.json, priv_curves.png.

The x axis of every curve is the private share r_p / (r_s + r_p) of the per-mode LoRA rank; SFT (no modes) is a
horizontal line. Rows marked "existing" are adapters of earlier runs evaluated with the same protocol (wta1 = share 0,
LoRA-MCL of phase1_seeds: 4 separate adapters without FiLM, trained 2 h by time, i.e. on fewer tokens).

math (hv, held-out hard problems):
  accuracy   pass@1 / pass@4 / pass@16 (modes as strata, zreport.pass_at), maj@16; paired bootstrap over problems
             against SFT of the same seed, pooled over seeds (the same resampled problems for every seed)
  greedy     one greedy solution per mode: accuracy, any mode right, distinct answers, phi of the errors of two modes
             (mean over pairs), the share of two wrong modes with the same answer
  loss       references of held-out val problems (and of training, if built): mean token NLL of the best mode and of
             the average mode against SFT (Table 1 of LoRA-MCL); which mode wins a reference: normalized mutual
             information with its strategy, source and length quartile, the best one-to-one match with the strategy,
             and P(two references of one problem have different winners) against two references of different problems
             (modes that split problems, not solutions, give a within-problem rate far below the across rate)
mt (newstest2014 en->de, 10 references): BLEU (first reference, all 10), oracle sentence BLEU, Pairwise-BLEU,
  leave-one-out BLEU, references covered; loss on the 10 references; the same within-sentence split of the winners."""
import argparse
import gzip
import itertools
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase0.answers import key as akey  # noqa: E402
from phase1.sreport import boot  # noqa: E402
from phase1.zreport import maj_at, pass_at  # noqa: E402


def read_gz(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(l) for l in f]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def jload(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return json.load(f)
    return json.load(open(path))


def f3(x, d=3):
    return "—" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def cell(t, d=3):
    return "—" if t is None else f"{t[0]:+.{d}f} [{t[1]:+.{d}f}, {t[2]:+.{d}f}]"


# ----------------------------------------------------------------------------- math: generations
def load_gen(gdir, probs):
    """C [S, n, N], A [S, n, N] answer index (-1 none), G [S, N] greedy correct, GA [S, N] greedy answer, truth [N];
    None if unfinished. ValueError if a file does not follow the set."""
    if not os.path.exists(os.path.join(gdir, "DONE")):
        return None
    S = json.load(open(os.path.join(gdir, "peval.json")))["modes"]
    N = len(probs)
    ids = [{} for _ in range(N)]
    truth = np.array([ids[j].setdefault(akey(p["answer"]), 0) for j, p in enumerate(probs)])

    def index(rows, f):
        if [r["id"] for r in rows] != [p["id"] for p in probs]:
            raise ValueError(f"{f}: the problems differ from the set")
        return np.array([[ids[j].setdefault(x, len(ids[j])) if x else -1 for x in r["answers"]]
                         for j, r in enumerate(rows)]).T, np.array([r["correct"] for r in rows], dtype=bool).T

    C, A, G, GA = [], [], [], []
    for m in range(S):
        f = os.path.join(gdir, f"solve_{m}_test.jsonl.gz")
        a_, c_ = index(read_gz(f), f)
        A.append(a_)
        C.append(c_)
        f = os.path.join(gdir, f"greedy_{m}_test.jsonl.gz")
        if os.path.exists(f):
            a_, c_ = index(read_gz(f), f)
            GA.append(a_[0])
            G.append(c_[0])
    return {"C": np.stack(C), "A": np.stack(A), "G": np.stack(G) if G else None, "GA": np.stack(GA) if GA else None,
            "truth": truth}


def pair_phi(W):
    """W [S, N] bool errors -> mean pairwise phi over mode pairs (pairs with a constant indicator skipped)."""
    v = []
    for i in range(len(W)):
        for j in range(i + 1, len(W)):
            x, y = W[i].astype(float), W[j].astype(float)
            if x.std() > 0 and y.std() > 0:
                v.append(np.corrcoef(x, y)[0, 1])
    return float(np.mean(v)) if v else float("nan")


def same_wrong(GA, G):
    both = same = 0
    for i in range(len(GA)):
        for j in range(i + 1, len(GA)):
            m = ~G[i] & ~G[j] & (GA[i] >= 0) & (GA[j] >= 0)
            both += m.sum()
            same += (GA[i][m] == GA[j][m]).sum()
    return float(same / both) if both else float("nan")


def math_vectors(E):
    """Per-problem vectors (bootstrap) and scalars."""
    C, A = E["C"], E["A"]
    S, n, N = C.shape
    rng = np.random.default_rng(0)
    v = {f"pass{k}": pass_at(C, k) for k in (1, 4, 16) if k <= S * n}
    v["maj_all"] = maj_at(A, E["truth"], S * n, rng)                 # majority of all S * n attempts
    v["distinct_all"] = np.array([len(set(A[:, :, j].ravel().tolist()) - {-1}) for j in range(N)], dtype=float)
    sc = {}
    if E["G"] is not None:
        G, GA = E["G"], E["GA"]
        v["greedy_acc"] = G.mean(0)
        v["greedy_any"] = G.any(0).astype(float)
        v["greedy_distinct"] = np.array([len(set(GA[:, j].tolist()) - {-1}) for j in range(N)], dtype=float)
        sc["greedy_phi"] = pair_phi(~G) if S > 1 else float("nan")
        sc["greedy_same_wrong"] = same_wrong(GA, G) if S > 1 else float("nan")
    sc["per_mode_acc"] = [float(C[m].mean()) for m in range(S)]
    return v, sc


# ----------------------------------------------------------------------------- loss on references
def nmi(x, y):
    """Normalized mutual information (arithmetic mean of the entropies) of two label vectors."""
    x, y = np.asarray([str(v) for v in x]), np.asarray([str(v) for v in y])     # None / mixed labels: categories
    if len(x) == 0:
        return float("nan")
    _, xi = np.unique(x, return_inverse=True)
    _, yi = np.unique(y, return_inverse=True)
    J = np.zeros((xi.max() + 1, yi.max() + 1))
    np.add.at(J, (xi, yi), 1)
    P = J / J.sum()
    px, py = P.sum(1), P.sum(0)
    nz = P > 0
    mi = float((P[nz] * np.log(P[nz] / np.outer(px, py)[nz])).sum())
    hx, hy = -float((px[px > 0] * np.log(px[px > 0])).sum()), -float((py[py > 0] * np.log(py[py > 0])).sum())
    return mi / ((hx + hy) / 2) if hx + hy > 0 else 0.0


def best_match(win, lab, K):
    """Share of references whose winning mode is the mode matched one-to-one to their label (best matching)."""
    labs = sorted(set(lab))
    if K > 6 or len(labs) > 6:
        return float("nan")
    best = 0
    for perm in itertools.permutations(range(max(K, len(labs))), len(labs)):
        m = dict(zip(labs, perm))
        best = max(best, sum(w == m[l] for w, l in zip(win, lab)))
    return best / len(win)


def split_rates(win, groups):
    """P(two references of one group have different winners) and the same for two references of different groups."""
    by = {}
    for w, g in zip(win, groups):
        by.setdefault(g, []).append(w)
    num = den = 0
    for ws in by.values():
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                num += ws[i] != ws[j]
                den += 1
    within = num / den if den else float("nan")
    cnt = np.bincount(np.asarray(win))
    n = len(win)
    tot_diff = (n * n - (cnt ** 2).sum()) / 2                     # pairs with different winners among all pairs
    tot = n * (n - 1) / 2
    across = (tot_diff - num) / (tot - den) if tot - den > 0 else float("nan")
    return within, across


def loss_stats(nll, ntok, rows=None, sft=None, groups=None):
    """nll [K, R] summed per reference, ntok [R]; rows: labels (strategy, source) for the winners."""
    nll, ntok = np.asarray(nll), np.asarray(ntok, dtype=float)
    per = nll / ntok                                       # [K, R] mean NLL per token of every reference
    # "min" / "mean": per reference first, then the mean over references (LoRA-MCL, Eq. 22); "_tok": all tokens pooled
    out = {"min": float(per.min(0).mean()), "mean": float(per.mean(0).mean()),
           "min_tok": float(nll.min(0).sum() / ntok.sum()), "mean_tok": float(nll.mean(0).sum() / ntok.sum()),
           "per_mode": per.mean(1).tolist()}
    if sft is not None:
        sft = np.asarray(sft, dtype=float)
        out["min_minus_sft"] = out["min"] - float((sft / ntok).mean())
        out["mean_minus_sft"] = out["mean"] - float((sft / ntok).mean())
        out["min_tok_minus_sft"] = out["min_tok"] - float(sft.sum() / ntok.sum())
    K = nll.shape[0]
    if K > 1:
        win = nll.argmin(0)
        sh = np.bincount(win, minlength=K) / len(win)
        out["win_share"] = sorted(sh.tolist(), reverse=True)     # sorted: mode indices mean nothing across seeds
        out["max_share"] = float(sh.max())
        if rows is not None:
            lab = [r["strategy"] for r in rows]
            q = np.quantile(ntok, [0.25, 0.5, 0.75])
            out["nmi_strategy"] = nmi(win, lab)
            out["match_strategy"] = best_match(win.tolist(), lab, K)
            out["nmi_source"] = nmi(win, [r.get("source") for r in rows])
            out["nmi_length"] = nmi(win, np.searchsorted(q, ntok))
        if groups is not None:
            out["split_within"], out["split_across"] = split_rates(win.tolist(), groups)
    return out


def aggregate(st):
    """Mean over seeds of every statistic a seed has (a statistic missing in some seeds averages the others);
    sorted win shares averaged position by position, the per-seed largest share kept."""
    keys = {k for x in st for k, v in x.items() if not isinstance(v, list)}
    avg = {k: float(np.nanmean([x.get(k, np.nan) for x in st])) for k in keys}
    ws = [x["win_share"] for x in st if x.get("win_share")]
    avg["win_share"] = np.mean(ws, 0).tolist() if ws else None
    avg["max_share_seeds"] = [x.get("max_share") for x in st]
    return avg


# ----------------------------------------------------------------------------- math report
def result_of(d):
    """result.json of a trained adapter ({} if missing): tokens, trainable_M."""
    f = d.get("result") or ""
    return json.load(open(f)) if f and os.path.exists(f) else {}


def math_report(run, P, boot_n):
    probs = load_jsonl(P["set_file"])
    refs = read_gz(P["refs"]) if P.get("refs") and os.path.exists(P["refs"]) else None
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(probs), (boot_n, len(probs)))
    V, SC, L, rej = {}, {}, {}, []
    for pt in P["points"]:
        for s, d in pt["seeds"].items():
            try:
                E = load_gen(d["gen"], probs)
            except (ValueError, OSError) as e:
                rej.append(str(e))
                E = None
            if E is not None:
                V[(pt["name"], s)], SC[(pt["name"], s)] = math_vectors(E)
            f = os.path.join(d["nll"], "nll.json.gz")
            if refs is not None and os.path.exists(f):
                L[(pt["name"], s)] = jload(f)
    names = [pt["name"] for pt in P["points"]]
    seeds = sorted({s for pt in P["points"] for s in pt["seeds"]}, key=int)
    lab = {pt["name"]: pt["label"] for pt in P["points"]}
    ms = ["pass1", "pass4", "pass16", "maj_all", "distinct_all", "greedy_acc", "greedy_any", "greedy_distinct"]
    k_all = P.get("k", 16)
    head = {"maj_all": f"maj@{k_all}", "distinct_all": f"разных ответов из {k_all}", "greedy_acc": "жадный: точность",
            "greedy_any": "жадный: хоть один прав", "greedy_distinct": "жадный: разных ответов",
            "pass1": "pass@1", "pass4": "pass@4", "pass16": "pass@16"}
    out = {"rows": {}, "delta_vs_sft": {}, "loss": {}, "rejected": rej}
    lines = ["# Развёртка доли приватной LoRA: математика (hv)\n",
             f"Набор: {P['set_file']} ({len(probs)} задач), {P.get('k', 16)} попыток на модель (по k/K на режим) + по "
             f"одному жадному решению на режим. Сиды: {', '.join(seeds)}.\n",
             "| точка | доля приватной | параметры, M | токены, M | " + " | ".join(head[m] for m in ms)
             + " | φ (жадн.) | same-wrong |",
             "|---|---|---|---|" + "---|" * (len(ms) + 2)]
    for pt in P["points"]:
        got = [s for s in seeds if (pt["name"], s) in V]
        if not got:
            lines.append(f"| {pt['label']} | {f3(pt.get('share'), 2)} | | | " + " | ".join("…" for _ in ms) + " | | |")
            continue
        row = {m: float(np.mean([V[(pt['name'], s)][m].mean() for s in got if m in V[(pt['name'], s)]]))
               if any(m in V[(pt['name'], s)] for s in got) else float("nan") for m in ms}
        row["greedy_phi"] = float(np.nanmean([SC[(pt["name"], s)].get("greedy_phi", np.nan) for s in got]))
        row["greedy_same_wrong"] = float(np.nanmean([SC[(pt["name"], s)].get("greedy_same_wrong", np.nan) for s in got]))
        row["seeds"] = got
        out["rows"][pt["name"]] = row
        rs = [result_of(pt["seeds"][s]) for s in got]
        tok = [r["tokens"] for r in rs if r.get("tokens")]
        par = [r["trainable_M"] for r in rs if r.get("trainable_M")]
        row["tokens_M"] = float(np.mean(tok)) / 1e6 if tok else float("nan")
        row["params_M"] = float(np.mean(par)) if par else float("nan")
        lines.append(f"| {pt['label']} | {f3(pt.get('share'), 2)} | {f3(row['params_M'], 1)} | {f3(row['tokens_M'], 1)} | "
                     + " | ".join(f3(row[m]) for m in ms) + f" | {f3(row['greedy_phi'])} | {f3(row['greedy_same_wrong'])} |")
    if "sft" in names:
        lines += ["", "## Разность с SFT того же сида (парный бутстрэп по задачам, сиды усреднены)\n",
                  f"| точка | pass@1 | pass@4 | pass@16 | maj@{k_all} | жадный: хоть один режим прав |",
                  "|---|---|---|---|---|---|"]
        for pt in P["points"]:
            if pt["name"] == "sft":
                continue
            ss = [s for s in seeds if (pt["name"], s) in V and ("sft", s) in V]
            if not ss:
                continue
            row, cells = {}, []
            for m in ("pass1", "pass4", "pass16", "maj_all", "greedy_any"):
                if all(m in V[(pt["name"], s)] and m in V[("sft", s)] for s in ss):
                    row[m] = boot(np.stack([V[(pt["name"], s)][m] - V[("sft", s)][m] for s in ss]), idx)
                cells.append(cell(row.get(m)))
            out["delta_vs_sft"][pt["name"]] = row
            lines.append(f"| {lab[pt['name']]} ({len(ss)} сид.) | " + " | ".join(cells) + " |")
    if refs is not None and L:
        for split in ("val", "train"):
            sel = [i for i, r in enumerate(refs) if r["split"] == split]
            if not sel:
                continue
            rows = [refs[i] for i in sel]
            groups = [r["id"] for r in rows]
            title = "отложенные (val, не обучались)" if split == "val" else "из обучения (ModC)"
            lines += ["", f"## Лосс на эталонах: {title}, {len(sel)} эталонов, {len(set(groups))} задач\n",
                      "Лосс — NLL на токен эталона, среднее по эталонам (как уравнение 22 LoRA-MCL); «по токенам» — "
                      "все токены вместе. «лучший» — минимум по режимам для каждого эталона. Доли побед отсортированы "
                      "(номера режимов разных сидов не связаны). «внутри» — P(два эталона одной задачи выигрывают "
                      "разные режимы), «между» — то же для разных задач.\n",
                      "| точка | лучший режим | средний режим | лучший − SFT | лучший − SFT, по токенам | доли побед | "
                      "NMI стратегия | совпадение со стратегией | NMI источник | NMI длина | внутри | между |",
                      "|---" * 12 + "|"]
            for pt in P["points"]:
                got = [s for s in seeds if (pt["name"], s) in L]
                if not got:
                    continue
                st = []
                keys = [f"{r['split']}|{r['id']}|{r['strategy']}" for r in refs]
                for s in got:
                    D = L[(pt["name"], s)]
                    if D.get("keys") != keys:
                        rej.append(f"{pt['name']} s{s}: the NLL file was computed on other references")
                        continue
                    nll = np.asarray(D["nll"])[:, sel]
                    ntok = np.asarray(D["ntok"])[sel]
                    paired = ("sft", s) in L and L[("sft", s)].get("keys") == keys
                    sft = np.asarray(L[("sft", s)]["nll"])[0, sel] if paired else None
                    st.append(loss_stats(nll, ntok, rows, sft, groups))
                if not st:
                    continue
                avg = aggregate(st)
                out["loss"].setdefault(split, {})[pt["name"]] = avg
                ws = "/".join(f"{w:.2f}" for w in avg["win_share"]) if avg["win_share"] else "—"
                lines.append(f"| {lab[pt['name']]} | {f3(avg['min'])} | {f3(avg['mean'])} | "
                             f"{f3(avg.get('min_minus_sft', np.nan))} | {f3(avg.get('min_tok_minus_sft', np.nan))} | "
                             f"{ws} | {f3(avg.get('nmi_strategy', np.nan))} | "
                             f"{f3(avg.get('match_strategy', np.nan))} | {f3(avg.get('nmi_source', np.nan))} | "
                             f"{f3(avg.get('nmi_length', np.nan))} | {f3(avg.get('split_within', np.nan))} | "
                             f"{f3(avg.get('split_across', np.nan))} |")
    return out, lines


# ----------------------------------------------------------------------------- mt report
def mt_report(run, P):
    test = load_jsonl(os.path.join(P["data"], "test.jsonl"))
    seeds = sorted({s for pt in P["points"] for s in pt["seeds"]}, key=int)
    M, L = {}, {}
    for pt in P["points"]:
        for s, d in pt["seeds"].items():
            f = os.path.join(d["eval"], "metrics.json")
            if os.path.exists(os.path.join(d["eval"], "DONE")) and os.path.exists(f):
                M[(pt["name"], s)] = json.load(open(f))
                g = os.path.join(d["eval"], "nll.json.gz")
                if os.path.exists(g):
                    L[(pt["name"], s)] = jload(g)
    beams = sorted({b for m in M.values() for b in m["beams"]}, key=int)
    keys = ["bleu_first_ref_mean", "bleu_all_refs_mean", "oracle_sentence_bleu", "pairwise_bleu", "pairwise_bleu_13a",
            "loo_bleu", "loo_bleu_13a", "refs_covered"]
    out = {"rows": {}, "loss": {}}
    lines = ["# Развёртка доли приватной LoRA: перевод en→de (newstest2014, 10 эталонов)\n",
             f"{len(test)} предложений. Метрики — порт compute_metrics.py LoRA-MCL (детокенизация Moses, sacreBLEU). "
             f"Pairwise-BLEU: ниже — разнообразнее; oracle и LOO-BLEU: выше — лучше. Сиды: {', '.join(seeds)}.\n"]
    for b in beams:
        lines += [f"## Луч {b}\n", "| точка | доля приватной | BLEU (1 эталон) | BLEU (10) | oracle BLEU | pairwise "
                  "(none) | pairwise (13a) | LOO-BLEU (none) | LOO-BLEU (13a) | покрыто эталонов |", "|---" * 10 + "|"]
        for pt in P["points"]:
            got = [s for s in seeds if (pt["name"], s) in M and b in M[(pt["name"], s)]["beams"]]
            if not got:
                lines.append(f"| {pt['label']} | {f3(pt.get('share'), 2)} |" + " … |" * len(keys))
                continue
            row = {k: float(np.mean([M[(pt["name"], s)]["beams"][b][k] for s in got])) for k in keys}
            row["seeds"] = got
            out["rows"].setdefault(b, {})[pt["name"]] = row
            lines.append(f"| {pt['label']} | {f3(pt.get('share'), 2)} | " + " | ".join(f3(row[k], 2) for k in keys) + " |")
    if L:
        lines += ["", "## Лосс на 10 эталонах\n", "«внутри» — P(два эталона одного предложения выигрывают разные режимы), "
                  "«между» — то же для разных предложений.\n",
                  "| точка | лучший режим | средний режим | лучший − SFT | доли побед | внутри | между |",
                  "|---|---|---|---|---|---|---|"]
        for pt in P["points"]:
            got = [s for s in seeds if (pt["name"], s) in L]
            if not got:
                continue
            st = []
            for s in got:
                D = L[(pt["name"], s)]
                ok = ("sft", s) in L and L[("sft", s)]["ref"] == D["ref"]
                sft = np.asarray(L[("sft", s)]["nll"])[0] if ok else None
                st.append(loss_stats(D["nll"], D["ntok"], None, sft, [r[0] for r in D["ref"]]))
            avg = aggregate(st)
            out["loss"][pt["name"]] = avg
            ws = "/".join(f"{w:.2f}" for w in avg["win_share"]) if avg["win_share"] else "—"
            lines.append(f"| {pt['label']} | {f3(avg['min'])} | {f3(avg['mean'])} | {f3(avg.get('min_minus_sft', np.nan))} "
                         f"| {ws} | {f3(avg.get('split_within', np.nan))} | {f3(avg.get('split_across', np.nan))} |")
    return out, lines


# ----------------------------------------------------------------------------- curves
def plot(P, out, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    if P["bench"] == "math":
        panels = [("pass1", "pass@1 (hv)", out["rows"]), ("pass16", "pass@16 (hv)", out["rows"]),
                  ("greedy_any", "жадный: хоть один режим прав", out["rows"]),
                  ("greedy_distinct", "разных жадных ответов", out["rows"])]
        loss = out["loss"].get("val", {})
        panels += [("min", "лосс лучшего режима (val)", loss), ("nmi_strategy", "NMI победитель ~ стратегия", loss)]
    else:
        b = sorted(out["rows"], key=int)[-1] if out["rows"] else None
        R = out["rows"].get(b, {})
        panels = [("bleu_first_ref_mean", f"BLEU, луч {b}", R), ("oracle_sentence_bleu", f"oracle BLEU, луч {b}", R),
                  ("pairwise_bleu", f"pairwise BLEU, луч {b}", R), ("loo_bleu", f"LOO-BLEU, луч {b}", R),
                  ("min", "лосс лучшего режима", out["loss"]), ("split_within", "внутри предложения: разные режимы", out["loss"])]
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (k, title, R) in zip(axs.ravel(), panels):
        xs, ys, other = [], [], []
        for pt in P["points"]:
            y = R.get(pt["name"], {}).get(k)
            if y is None or not np.isfinite(y):
                continue
            if pt["name"] == "sft":
                ax.axhline(y, ls="--", color="gray", label="SFT")
            elif pt.get("curve", True):
                xs.append(pt["share"])
                ys.append(y)
            else:
                other.append((pt["share"], y, pt["label"]))
        o = np.argsort(xs)
        ax.plot(np.array(xs)[o], np.array(ys)[o], "o-", label="общая + приватная LoRA, FiLM")
        for x, y, l in other:
            ax.plot([x], [y], "s", label=l)
        ax.set_title(title)
        ax.set_xlabel("доля приватной LoRA")
        ax.grid(alpha=0.3)
    axs.ravel()[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--mark", default="")
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    P = json.load(open(os.path.join(a.run, "points.json")))
    out, lines = math_report(a.run, P, a.boot) if P["bench"] == "math" else mt_report(a.run, P)
    png = os.path.join(a.run, "priv_curves.png")
    if plot(P, out, png):
        lines += ["", f"Кривые: {png}"]
    if out.get("rejected"):
        lines += ["", "## Отклонённые файлы\n"] + [f"- {r}" for r in out["rejected"]]
    lines += ["", "## Точки\n"] + [f"- **{pt['label']}**: {pt.get('note', '')}" for pt in P["points"]]
    with open(os.path.join(a.run, "report_priv.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    json.dump(out, open(os.path.join(a.run, "summary_priv.json"), "w"), indent=1, default=float)
    print("\n".join(lines))
    if a.mark:
        os.makedirs(a.mark, exist_ok=True)
        open(os.path.join(a.mark, "DONE"), "w").close()


if __name__ == "__main__":
    main()
