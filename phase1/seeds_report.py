"""Report over seeds: mean +- SD across seeds and paired bootstrap against the reference methods.

Bootstrap: problems are resampled (the same indices for all methods); one replicate of a comparison is
the mean over seeds of X(seed) - R(seed), so pairing holds by problem and by seed. The interval covers
the variation of problems, the SD column covers the variation of seeds."""
import argparse
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase0.report import fmt, load_methods, phi_within, same_wrong  # noqa: E402
from phase1.metrics import load_eval  # noqa: E402
from pilot.metrics import auroc, majority  # noqa: E402

VARIANTS = [("wta1", "WTA, 1 эталон"), ("wtabal", "WTA, несколько эталонов + паросочетание"),
            ("sftmulti", "SFT, равное время"), ("sftmulti_eq", "SFT, токенов столько же, сколько у wta1"),
            ("loramcl", "LoRA-MCL: 4 отдельные LoRA, relaxed WTA"), ("modc", "ModC: префикс + стратегия эталона")]
OLD = ("wta1", "wtabal", "sftmulti")
REFS = ["sftmulti", "sftmulti_eq", "p0_strategies"]
METRICS = [("pass1", "pass@1"), ("maj4", "maj@4"), ("pass4", "pass@4"), ("phi", "φ ошибок"),
           ("same_wrong", "одинаковый неверный")]
P0 = "p0_strategies"


def eval_dir(run, phase1, v, s):
    return os.path.join(phase1, "eval", v) if phase1 and s == 0 and v in OLD else os.path.join(run, "eval", f"{v}_s{s}")


def vectors(A):
    cor, ans = A["correct"][:, 0], A["answer"][:, 0]
    return {"pass1": cor.mean(0), "maj4": (majority(ans) == A["truth"]).astype(float), "pass4": cor.any(0).astype(float)}


def metric(A, vec, k, idx):
    if k == "phi":
        return phi_within(~A["correct"][:, 0], A["op"], idx)
    if k == "same_wrong":
        return same_wrong(A, idx)
    return float(vec[k][idx].mean())


def verifier_auroc(A):
    """Mean AUROC of Yes/No over cells: a mode on its own solutions / on the solutions of other modes."""
    p, err = A["p_err"], A["has_err"][:, 0]
    if not np.isfinite(p).all():
        return float("nan"), float("nan")
    M = p.shape[0]
    cell = np.array([[auroc(p[i, j], err[i]) for j in range(M)] for i in range(M)])
    with np.errstate(all="ignore"):
        return float(np.nanmean(np.diag(cell))), float(np.nanmean(cell[~np.eye(M, dtype=bool)]))


def load_all(run, phase1, phase0, test):
    """methods[name][seed] = A; a phase-0 method has the single seed None."""
    methods, missing = {}, []
    if phase0 and os.path.isdir(phase0):
        tests, m0 = load_methods(phase0)
        if "strategies" in m0 and len(tests) == len(test) and [t["id"] for t in tests] == [t["id"] for t in test]:
            methods[P0] = {None: m0["strategies"][0]}
        else:
            missing.append("стратегии фазы 0")
    for v, _ in VARIANTS:
        for s in (0, 1, 2):
            r = load_eval(eval_dir(run, phase1, v, s), test, "test")
            if r is None:
                missing.append(f"{v} seed {s}")
            else:
                methods.setdefault(v, {})[s] = r[0]
    return methods, missing


def seed_pairs(methods, X, R):
    if R == P0:
        return [(s, None) for s in sorted(methods[X])]
    return [(s, s) for s in sorted(methods[X]) if s in methods[R]]


def compare(point, reps, X, R, pairs, k):
    """Difference X - R: mean and SD over seeds, 95% interval of the mean over seeds by problems."""
    per_seed = np.array([point[X, a][k] - point[R, b][k] for a, b in pairs])
    ok = per_seed[np.isfinite(per_seed)]
    if not len(ok):
        return None
    rep = np.nanmean(np.stack([reps[X, a][k] - reps[R, b][k] for a, b in pairs]), axis=0)
    rep = rep[np.isfinite(rep)]
    m = float(ok.mean())
    lo, hi = np.percentile(rep, [2.5, 97.5]) if len(rep) else (float("nan"), float("nan"))
    return {"mean": m, "sd": float(ok.std(ddof=1)) if len(ok) > 1 else 0.0, "lo": float(lo), "hi": float(hi),
            "agree": int((np.sign(ok) == np.sign(m)).sum()), "n": len(ok),
            "per_seed": {str(a): float(v) for (a, _), v in zip(pairs, per_seed)}}


def cell(d):
    if not d:
        return "—"
    sd = f" ±{d['sd']:.3f}" if d["n"] > 1 else ""
    return f"{d['mean']:+.3f}{sd} [{d['lo']:+.3f}, {d['hi']:+.3f}] {d['agree']}/{d['n']}"


def train_rows(run, phase1):
    rows = []
    for v, _ in VARIANTS:
        for s in (0, 1, 2):
            if v == "sftmulti_eq":
                f = os.path.join(run, "train", "sftmulti_eq_s0" if s == 0 else f"sftmulti_s{s}", "snap", "snap.json")
            elif phase1 and s == 0 and v in OLD:
                f = os.path.join(phase1, "train", v, "result.json")
            else:
                f = os.path.join(run, "train", f"{v}_s{s}", "result.json")
            if os.path.exists(f):
                rows.append((v, s, json.load(open(f))))
    return rows


def build(run, data, phase1, phase0, boot=1000):
    warnings.filterwarnings("ignore", category=RuntimeWarning)     # means over bootstrap samples without errors
    test = [json.loads(l) for l in open(os.path.join(data, "test.jsonl"))]
    N = len(test)
    methods, missing = load_all(run, phase1, phase0, test)
    rng = np.random.default_rng(0)
    idxs = [rng.integers(0, N, N) for _ in range(boot)]
    full = np.arange(N)
    point, reps, vers = {}, {}, {}
    for name, by_seed in methods.items():
        for s, A in by_seed.items():
            vec = vectors(A)
            point[name, s] = {k: metric(A, vec, k, full) for k, _ in METRICS}
            reps[name, s] = {k: np.array([metric(A, vec, k, i) for i in idxs]) for k, _ in METRICS}
            vers[name, s] = verifier_auroc(A)

    L = ["# Фаза 1, повторы с seed и сравнение с LoRA-MCL и ModC\n"]
    L.append(f"- задач в тесте: {N}. Seed 0 для wta1, wtabal, sftmulti взят из фазы 1 без перезапуска.")
    if missing:
        L.append(f"- **нет данных:** {', '.join(missing)}")
    sp = os.path.join(run, "state.json")
    if os.path.exists(sp):
        st = json.load(open(sp))
        bad = {k: v["status"] for k, v in st.get("jobs", {}).items() if v.get("status") in ("failed", "skipped")}
        if bad:
            L.append(f"- **не выполнены:** {bad} (см. logs/)")
        if st.get("eq_tokens"):
            L.append(f"- порог sftmulti_eq: {st['eq_tokens'] / 1e6:.1f}M токенов (wta1, seed 0).")

    L.append("\n## 1. Среднее ± SD по seed\n")
    L.append("| метод | seed | " + " | ".join(n for _, n in METRICS) + " | AUROC Yes/No: свой / чужой |")
    L.append("|---|---|" + "---|" * (len(METRICS) + 1))
    summary = {}
    for name, desc in VARIANTS + [(P0, "стратегии через промпт, фаза 0 (один прогон)")]:
        if name not in methods:
            continue
        seeds = sorted(methods[name], key=str)
        row, summ = [], {}
        for k, _ in METRICS:
            x = np.array([point[name, s][k] for s in seeds])
            m = float(np.nanmean(x))
            sd = float(np.nanstd(x, ddof=1)) if np.isfinite(x).sum() > 1 else float("nan")
            summ[k] = {"mean": m, "sd": sd, "per_seed": {str(s): float(v) for s, v in zip(seeds, x)}}
            row.append(fmt(m) + (f" ± {sd:.3f}" if np.isfinite(sd) else ""))
        va = np.array([vers[name, s] for s in seeds])
        av = [float(np.nanmean(va[:, i])) for i in (0, 1)]
        summ["auroc_own"], summ["auroc_other"] = av
        summary[name] = summ
        L.append(f"| {name} ({desc}) | {', '.join(str(s) for s in seeds) if seeds != [None] else '—'} | "
                 + " | ".join(row) + f" | {fmt(av[0])} / {fmt(av[1])} |")

    L.append("\n## 2. Парные сравнения, X − R\n")
    L.append("Ячейка: среднее по seed ± SD разностей по seed [95% бутстрап по задачам] и число seed, где знак "
             "разности совпал со средним. Для pass@1, maj@4, pass@4 лучше больше, для φ и одинакового неверного "
             "ответа — меньше.\n")
    comps = {}
    for R in REFS:
        if R not in methods:
            continue
        L.append(f"\n### Против {R}\n")
        L.append("| X | " + " | ".join(n for _, n in METRICS) + " |")
        L.append("|---|" + "---|" * len(METRICS))
        for X, _ in VARIANTS:
            if X == R or X not in methods or (R == "sftmulti" and X == "sftmulti_eq"):
                continue
            pairs = seed_pairs(methods, X, R)
            if not pairs:
                continue
            cells = []
            for k, _n in METRICS:
                d = compare(point, reps, X, R, pairs, k)
                comps[f"{X}-{R}:{k}"] = d
                cells.append(cell(d))
            L.append(f"| {X} | " + " | ".join(cells) + " |")

    L.append("\n## 3. Обучение\n")
    L.append("| вариант | seed | шагов | эпох | токенов | часов | параметров | доля решений по режимам | LR снимка |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    tr = train_rows(run, phase1)
    for v, s, r in tr:
        share = r.get("mode_share")
        L.append(f"| {v} | {s} | {r.get('steps_done', r.get('step'))} | {r.get('epochs_done', r.get('epochs', 0)):.2f} | "
                 f"{r['tokens'] / 1e6:.1f}M | {r['hours']:.2f} | "
                 f"{str(round(r['trainable_M'], 1)) + 'M' if r.get('trainable_M') else '—'} | "
                 f"{[round(x, 2) for x in share] if share else '—'} | "
                 f"{'×' + format(r['lr_mult'], '.2f') if 'lr_mult' in r else '—'} |")
    ref = {s: r["tokens"] for v, s, r in tr if v == "wta1"}
    eq = {s: r["tokens"] for v, s, r in tr if v == "sftmulti_eq"}
    off = {s: round(eq[s] / ref[s], 2) for s in eq if s in ref and abs(eq[s] / ref[s] - 1) > 0.1}
    if off:
        L.append(f"\n**Токены sftmulti_eq расходятся с wta1 больше чем на 10%** (отношение по seed): {off}.")
    L.append("\n## Как читать\n")
    L.append("- Устойчивость: смотрите SD и число совпавших знаков в разделе 2.\n"
             "- Переобучение SFT: сравните столбцы «против sftmulti» и «против sftmulti_eq». Если выигрыш wta1 и wtabal "
             "держится против sftmulti_eq, он не объясняется лишними проходами SFT по данным.\n"
             "- sftmulti_eq — снимок посреди косинусного расписания 2-часового запуска (LR снимка в разделе 3), "
             "а wta1 доходит до конца расписания; это не полностью равные условия.\n"
             "- LoRA-MCL: 4 адаптера r = 32, то есть в 4 раза больше параметров, и каждый пример проходит через все 4, "
             "поэтому за 2 часа шагов меньше (раздел 3).\n"
             "- ModC: префикс в начале входа, разбиение по стратегии промпта, которым получен эталон.\n"
             "- Если в «доле решений по режимам» один режим близок к 1, режимы схлопнулись.")
    out = os.path.join(run, "report_seeds.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    json.dump({"summary": summary, "comparisons": comps, "missing": missing},
              open(os.path.join(run, "summary_seeds.json"), "w"), default=float, indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--phase1", default="")
    ap.add_argument("--phase0", default="")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--mark", default="")
    a = ap.parse_args()
    print(build(a.run, a.data, a.phase1, a.phase0, a.boot))
    if a.mark:
        os.makedirs(a.mark, exist_ok=True)
        open(os.path.join(a.mark, "DONE"), "w").close()
