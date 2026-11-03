"""Report of multi-round self-training (phase1/srun.py) -> report_self.md, summary_self.json, audit_sample.md.

Primary endpoint, fixed before the run: pass@16 on hv (held-out hard problems) of the final model M_R, wta1 - rftbal,
paired bootstrap over problems, per seed and pooled over seeds (the same problems resampled for every seed). wta1 is
called better than X when the pooled 95% interval is above 0, every seed agrees in sign and the difference is at
least MIN_EFFECT. Everything else (pass@1 / pass@4, test, other baselines, discoveries) is secondary and exploratory.

Estimators: pass@k without replacement within a stratum (zreport.pass_at): the 4 modes are strata, k / 4 attempts
from each (pass@4 = one attempt of every mode), a model without modes is one stratum. maj@4: zreport.maj_at.
A discovery is an admitted solution (right answer with \\boxed{}), the rule of the bank."""
import argparse
import gzip
import json
import os
import random
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase0.answers import key as akey  # noqa: E402
from phase0.answers import last_boxed  # noqa: E402
from phase0.report import fmt  # noqa: E402
from phase1.srun import START  # noqa: E402
from phase1.zreport import maj_at, pass_at  # noqa: E402

MIN_EFFECT = 0.02
NAMES = {"rft": "RFT (sftmulti, вес 1 на решение)", "rftbal": "RFT, вес задачи 1 (sftmulti)",
         "wta1": "wta1: 4 кода FiLM, 1/\\|W\\|", "loramcl": "LoRA-MCL: 4 LoRA, 1/\\|W\\|",
         "swap": "sftmulti на банке wta1 (вес задачи 1)"}
HARD = ("A", "B0", "B12")                  # held-out hard strata
HARD_POOL = HARD + ("Av",)                 # + 0-correct questions seen as verifier examples: pool only


def read_gz(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(l) for l in f]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def load(gdir, name, probs):
    """dict(C correct, B admitted, A answer index (-1 none), X has \\boxed: [S, n, N]; truth [N]) or None."""
    if not os.path.exists(os.path.join(gdir, "DONE")):
        return None
    M = json.load(open(os.path.join(gdir, "gen.json")))["modes"]
    if not os.path.exists(os.path.join(gdir, f"solve_0_{name}.jsonl.gz")):
        return None
    N = len(probs)
    ids = [{} for _ in range(N)]
    truth = np.array([ids[j].setdefault(akey(p["answer"]), 0) for j, p in enumerate(probs)])
    C, A, X = [], [], []
    for m in range(max(M, 1)):
        rows = read_gz(os.path.join(gdir, f"solve_{m}_{name}.jsonl.gz"))
        if [r["id"] for r in rows] != [p["id"] for p in probs]:
            raise ValueError(f"{gdir}/solve_{m}_{name}: the problems differ from {name}.jsonl")
        C.append(np.array([r["correct"] for r in rows], dtype=bool).T)
        A.append(np.array([[ids[j].setdefault(x, len(ids[j])) if x else -1 for x in r["answers"]]
                           for j, r in enumerate(rows)]).T)
        X.append(np.array([[last_boxed(t) is not None for t in r["texts"]] for r in rows], dtype=bool).T)
    C, A, X = np.stack(C), np.stack(A), np.stack(X)
    return {"C": C, "B": C & X, "A": A, "X": X, "truth": truth}


def vectors(E, rng, ks=(1, 4, 16)):
    S, n, N = E["C"].shape
    v = {f"pass{k}": pass_at(E["C"], k) for k in ks if k // S + (k % S > 0) <= n}
    v["maj4"] = maj_at(E["A"], E["truth"], 4, rng, draws=50)
    first4 = E["A"][:, :4 // S].reshape(-1, N) if S > 1 else E["A"][0, :4]
    v["distinct4"] = np.array([len(set(first4[:, j].tolist())) for j in range(N)], dtype=float)
    v["noboxed"] = 1 - E["X"].reshape(-1, N).mean(0)
    return v


def complementarity(E):
    """Modes with 4 attempts each: union of one attempt per mode against 4 attempts of a single mode."""
    C = E["C"]
    S, n, N = C.shape
    if S < 2 or n < 4:
        return None
    within = float(np.mean([C[m].any(0).mean() for m in range(S)]))
    bal = float(pass_at(C, S).mean())
    solved = C.any(1)                                                        # [S, N]
    excl = (solved.sum(0) == 1) & solved.any(0)
    jac = [float((solved[i] & solved[j]).sum() / max(1, (solved[i] | solved[j]).sum()))
           for i in range(S) for j in range(i + 1, S)]
    return {"balanced4": bal, "within4": within, "benefit": bal - within,
            "exclusive": float(excl.sum() / max(1, solved.any(0).sum())), "jaccard": float(np.mean(jac)),
            "per_mode_acc": [float(C[m].mean()) for m in range(S)]}


def boot(dv, idxs):
    """dv: per-problem differences, [N] or [seeds, N] (averaged over seeds for the same resampled problems)."""
    dv = np.atleast_2d(dv).mean(0)
    reps = dv[idxs].mean(1)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return float(dv.mean()), float(lo), float(hi)


def cell(t):
    return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def build(run, seeds, chains, R, boot_n=1000):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    dd = os.path.join(run, "data")
    P = {n: load_jsonl(os.path.join(dd, f"{n}.jsonl")) for n in ("pool", "hv", "frozen", "test")}
    meta = json.load(open(os.path.join(dd, "meta.json")))
    rng = np.random.default_rng(0)
    swap = os.path.isdir(os.path.join(run, f"s{seeds[0]}", "swap"))
    allc = list(chains) + (["swap"] if swap else [])
    evr = sorted({0, R} | {r for r in range(2, R, 2)})

    def gdir(s, c, r):
        if c == "swap":
            return os.path.join(run, f"s{s}", "swap", "gen")
        return os.path.join(run, f"s{s}", f"start_{START[c]}", "gen") if r == 0 else \
            os.path.join(run, f"s{s}", c, f"r{r}", "gen")

    V, E, rejected, cache = {}, {}, [], {}         # (s, c, r, set) -> vectors / arrays; cache by file
    for s in seeds:
        for c in allc:
            for r in ([R] if c == "swap" else range(R + 1)):
                for name in ("hv", "test", "pool", "frozen"):
                    try:
                        e = load(gdir(s, c, r), name, P[name])
                    except (ValueError, OSError) as ex:
                        rejected.append(str(ex))
                        continue
                    if e is None:
                        continue
                    E[(s, c, r, name)] = e
                    if name in ("hv", "test"):        # shared round 0 of rft / rftbal: the same numbers
                        k = (gdir(s, c, r), name)
                        if k not in cache:
                            cache[k] = vectors(e, np.random.default_rng(0))
                        V[(s, c, r, name)] = cache[k]
    done = {s: [c for c in allc if (s, c, R, "hv") in V] for s in seeds}
    done_seeds = {c: [s for s in seeds if c in done[s]] for c in allc}
    full = [s for s in seeds if all(c in done[s] for c in chains)]
    N_hv = len(P["hv"])
    idx_hv = rng.integers(0, N_hv, (boot_n, N_hv))
    idx_te = rng.integers(0, len(P["test"]), (boot_n, len(P["test"])))
    comps = {}

    def compare(X, Y, r, name, m, seeds_, per_ci=False):
        """(pooled (mean, lo, hi), per-seed means or (mean, lo, hi), the seeds used)."""
        ss = [s for s in seeds_ if (s, X, r, name) in V and (s, Y, r, name) in V and m in V[(s, X, r, name)]
              and m in V[(s, Y, r, name)]]
        if not ss:
            return None, [], []
        idx = idx_hv if name == "hv" else idx_te
        dv = np.stack([V[(s, X, r, name)][m] - V[(s, Y, r, name)][m] for s in ss])
        per = [boot(d, idx) if per_ci else float(d.mean()) for d in dv]
        return boot(dv, idx), per, ss

    L = ["# Многораундовое самообучение: открывает ли модель новые задачи\n"]
    L.append(f"- seeds: {seeds} (все цепочки закончены: {full or 'нет'}); раундов {R}; задачи: пул {meta['sizes']['pool']}"
             f" (трудные {sum(meta['pool'].get(k, 0) for k in HARD_POOL)}, средние {meta['pool'].get('B36', 0)}), "
             f"отложенные трудные hv {meta['sizes']['hv']}, frozen {meta['sizes']['frozen']}, тест "
             f"{meta['sizes']['test']}.")
    L.append(f"- страты трудных: A — 0 верных решений базовой модели из 8–16 в старых пулах, в обучении фазы 1 их не "
             f"было; Av — то же, но вопрос был в примерах проверки как «No» (только в пуле, не в hv); B0 / B12 — новые "
             f"задачи, 0 / 1–2 верных из 8 у базовой модели; B36 — новые средние (3–6 из 8), только в пуле. "
             f"hv: {meta['hv']}, пул: {meta['pool']}.")
    cj = os.path.join(run, "pool", "cands.json")
    if os.path.exists(cj):
        c = json.load(open(cj))
        L.append(f"- кандидаты: A {c['A']}, B {c['B']} {c['B_by_source']}" + (f"; **{c['notes']}**" if c["notes"] else ""))
    st = json.load(open(os.path.join(run, "state.json"))) if os.path.exists(os.path.join(run, "state.json")) else {}
    bad = {j: x["status"] for j, x in st.get("jobs", {}).items() if x.get("status") in ("failed", "skipped")}
    if bad:
        L.append(f"- **не выполнены:** {bad} (см. logs/)")
    missing = [f"s{s}:{c}" for s in seeds for c in allc if c not in done[s]]
    if missing:
        L.append(f"- **нет финальной модели:** {', '.join(missing)}")
    if rejected:
        L.append(f"- **отклонены файлы:** {rejected}")

    # ------------------------------------------------------------------ 1. primary
    L.append(f"\n## 1. Главная метрика (зафиксирована до запуска): pass@16 на hv у M_{R}, wta1 − rftbal\n")
    L.append(f"Решение «wta1 лучше X»: общий по seed 95% интервал выше 0, у каждого seed разность > 0 и общая разность "
             f"≥ {MIN_EFFECT}. Интервал — бутстрап по задачам при этих обучениях; разброс между обучениями он не "
             f"учитывает, его видно только по столбцу seed.\n")
    L.append("| сравнение | разность [95%] по всем seed | по seed [95%] | вывод |")
    L.append("|---|---|---|---|")
    for Y in [c for c in ("rftbal", "rft", "loramcl") if c in chains]:
        t, per, ss = compare("wta1", Y, R, "hv", "pass16", seeds, per_ci=True) if "wta1" in chains else (None, [], [])
        if t is None:
            L.append(f"| wta1 − {Y} | — | — | нет данных |")
            continue
        win = t[1] > 0 and all(p[0] > 0 for p in per) and t[0] >= MIN_EFFECT
        lack = [s for s in seeds if s not in ss]
        verdict = ("**wta1 лучше**" if win else "не показано") + (
            f" (предварительно: нет seed {lack})" if lack else "")
        comps[f"primary:wta1-{Y}"] = {"mean": t[0], "lo": t[1], "hi": t[2], "per_seed": dict(zip(ss, per)),
                                      "missing_seeds": lack, "win": bool(win and not lack)}
        L.append(f"| wta1 − {Y}{' (главное)' if Y == 'rftbal' else ''} | {cell(t)} | "
                 f"{'; '.join(f's{x}: {cell(p)}' for x, p in zip(ss, per))} | {verdict} |")

    # ------------------------------------------------------------------ 2. hv
    def cohort(c):
        """The seeds where chain c reached its final model (tables average over the same seeds at every round)."""
        return done_seeds.get(c) or []

    def mean_over(c, r, name, m, ss=None):
        xs = [V[(s, c, r, name)][m].mean() for s in (cohort(c) if ss is None else ss)
              if (s, c, r, name) in V and m in V[(s, c, r, name)]]
        return float(np.mean(xs)) if xs else None

    L.append(f"\n## 2. hv: отложенные трудные задачи по раундам\n")
    L.append("Среднее по тем seed, где цепочка дошла до последней модели (они в скобках), одинаковое для всех "
             "раундов; изменение M_R − M_0 считается внутри seed.\n")
    cols = [(r, m) for r in evr for m in (("pass1", "pass4", "pass16", "maj4") if r in (0, R) else ("pass1", "pass4"))]
    L.append("| цепочка | " + " | ".join(f"M_{r} {m.replace('pass', 'pass@').replace('maj4', 'maj@4')}"
                                       for r, m in cols) + f" | Δ pass@16 (M_{R} − M_0) |")
    L.append("|---|" + "---|" * (len(cols) + 1))
    sft = next((c for c in ("rftbal", "rft") if c in chains), None)      # M_0 of swap: the start sftmulti
    for c in allc:
        ss = cohort(c)
        vals = [mean_over(c, r, "hv", m) if c != "swap" or r == R else None for r, m in cols]
        if c == "swap" and sft:
            vals = [mean_over(sft, 0, "hv", m, ss) if r == 0 else v for (r, m), v in zip(cols, vals)]
        src = sft if c == "swap" else c
        ch = [V[(s, c, R, "hv")]["pass16"].mean() - V[(s, src, 0, "hv")]["pass16"].mean() for s in ss
              if src and (s, src, 0, "hv") in V and "pass16" in V[(s, src, 0, "hv")]]
        L.append(f"| {c} ({NAMES[c]}; seed {ss or 'нет'}) | " + " | ".join(fmt(x) for x in vals) +
                 f" | {fmt(float(np.mean(ch))) if ch else '—'} |")

    L.append(f"\nИзменение за самообучение, разность разностей (M_{R} − M_0)_wta1 − (M_{R} − M_0)_X на hv:\n")
    L.append("| сравнение | pass@1 | pass@4 | pass@16 |")
    L.append("|---|---|---|---|")
    for Y in [c for c in ("rftbal", "rft", "loramcl") if c in chains and "wta1" in chains]:
        cells = []
        for m in ("pass1", "pass4", "pass16"):
            ss = [s for s in seeds if all((s, c, r, "hv") in V for c in ("wta1", Y) for r in (0, R))]
            if not ss:
                cells.append("—")
                continue
            dv = np.stack([(V[(s, "wta1", R, "hv")][m] - V[(s, "wta1", 0, "hv")][m]) -
                           (V[(s, Y, R, "hv")][m] - V[(s, Y, 0, "hv")][m]) for s in ss])
            t = boot(dv, idx_hv)
            comps[f"change:wta1-{Y}:{m}"] = {"mean": t[0], "lo": t[1], "hi": t[2]}
            cells.append(cell(t))
        L.append(f"| wta1 − {Y} | " + " | ".join(cells) + " |")

    L.append(f"\npass@16 на hv у M_{R} по стратам:\n")
    L.append("| цепочка | " + " | ".join(HARD) + " |")
    L.append("|---|" + "---|" * len(HARD))
    strat = np.array([p.get("stratum") for p in P["hv"]])
    for c in allc:
        row = []
        for h in HARD:
            msk = strat == h
            xs = [V[(s, c, R, "hv")]["pass16"][msk].mean() for s in cohort(c) if msk.any()]
            row.append(fmt(float(np.mean(xs))) + f" (n={int(msk.sum())})" if xs else "—")
        L.append(f"| {c} | " + " | ".join(row) + " |")

    # ------------------------------------------------------------------ 3. test
    L.append(f"\n## 3. Тест фазы 1 (4 попытки на задачу; seed — как в разделе 2)\n")
    cols = [(r, m) for r in evr for m in ("pass1", "pass4", "maj4")]
    L.append("| цепочка | " + " | ".join(f"M_{r} {m.replace('pass', 'pass@').replace('maj4', 'maj@4')}"
                                       for r, m in cols) + " |")
    L.append("|---|" + "---|" * len(cols))
    for c in allc:
        L.append(f"| {c} | " + " | ".join(fmt(mean_over(c, r, "test", m)) if c != "swap" or r == R else
                                         fmt(mean_over(sft, 0, "test", m, cohort(c))) if r == 0 and sft else "—"
                                         for r, m in cols) + " |")

    # ------------------------------------------------------------------ 4. secondary comparisons
    L.append(f"\n## 4. Где wta1 лучше: все заранее перечисленные вторичные метрики у M_{R}\n")
    L.append("Разведка: метрик и сравнений много, отдельные «выигрыши» без поправки на множественность — повод для "
             "следующего опыта, а не вывод.\n")
    L.append("| метрика | " + " | ".join(f"wta1 − {Y}" for Y in ("rftbal", "rft", "loramcl") if Y in chains) + " |")
    L.append("|---|" + "---|" * sum(Y in chains for Y in ("rftbal", "rft", "loramcl")))
    for name, m in (("hv", "pass1"), ("hv", "pass4"), ("hv", "pass16"), ("hv", "maj4"), ("test", "pass1"),
                    ("test", "pass4"), ("test", "maj4")):
        cells = []
        for Y in [c for c in ("rftbal", "rft", "loramcl") if c in chains]:
            t, per, _ = compare("wta1", Y, R, name, m, seeds) if "wta1" in chains else (None, [], [])
            if t is None:
                cells.append("—")
                continue
            comps[f"{name}:{m}:wta1-{Y}"] = {"mean": t[0], "lo": t[1], "hi": t[2], "per_seed": per}
            mark = " ✓" if t[1] > 0 and all(p > 0 for p in per) else " ✗" if t[2] < 0 and all(p < 0 for p in per) else ""
            cells.append(cell(t) + mark)
        L.append(f"| {name} {m.replace('pass', 'pass@').replace('maj4', 'maj@4')} | " + " | ".join(cells) + " |")
    L.append("\n✓ — интервал выше 0 и все seed > 0; ✗ — ниже 0 и все seed < 0.")

    # ------------------------------------------------------------------ 5. acquisition
    L.append(f"\n## 5. Открытия на пуле: трудные задачи пула, решённые в раунде (покрытие 4 попытками) / хотя бы раз "
             f"к этому раунду\n")
    L.append("Решённой считается задача с допущенным в банк решением (верный ответ в \\boxed{}). Накопленное число "
             "растёт и без обучения: строка frozen — та же стартовая модель без обучения, 4·(r+1) попыток на тех же "
             f"{meta['sizes']['frozen']} задачах frozen (несмещённая оценка по 24 попыткам). Сравнивать с ней нужно "
             "столбец «frozen-задачи» цепочки.\n")
    hard_pool = np.array([p.get("stratum") in HARD_POOL for p in P["pool"]])
    fz = {p["id"] for p in P["frozen"]}
    in_fz = np.array([p["id"] in fz for p in P["pool"]])
    L.append("| цепочка | " + " | ".join(f"раунд {r}" for r in range(R)) + " | frozen-задачи, хотя бы раз |")
    L.append("|---|" + "---|" * (R + 1))
    acq = {}
    for c in chains:
        cells, cumf = [], []
        for r in range(R):
            xs, cs, fs = [], [], []
            for s in [s for s in seeds if all((s, c, q, "pool") in E for q in range(R))]:   # chains with all rounds
                es = [E.get((s, c, q, "pool")) for q in range(r + 1)]
                if any(e is None for e in es):
                    continue
                now = es[-1]["B"].any((0, 1))
                ever = np.any([e["B"].any((0, 1)) for e in es], axis=0)
                xs.append(now[hard_pool].mean())
                cs.append(ever[hard_pool].mean())
                fs.append(ever[in_fz].mean() if in_fz.any() else np.nan)
            if xs:
                acq[f"{c}:r{r}"] = {"cover": float(np.mean(xs)), "ever": float(np.mean(cs)), "frozen_ids": float(np.mean(fs))}
            cells.append(f"{np.mean(xs):.3f} / {np.mean(cs):.3f}" if xs else "—")
            cumf.append(f"{np.mean(fs):.3f}" if xs else "—")
        L.append(f"| {c} | " + " | ".join(cells) + f" | {' → '.join(cumf)} |")
    if [p["id"] for p in P["pool"] if p["id"] in fz] != [p["id"] for p in P["frozen"]]:
        raise ValueError("frozen.jsonl is not the pool subset in pool order")
    for stt in sorted({START[c] for c in chains}):
        c0 = next(c for c in chains if START[c] == stt)
        curve = []
        for j in range(1, R + 1):
            xs = []
            for s in seeds:
                e0, ef = E.get((s, c0, 0, "pool")), E.get((s, c0, 0, "frozen"))
                if e0 is None or ef is None:
                    continue
                Bc = np.concatenate([e0["B"][:, :, in_fz], ef["B"]], axis=1)       # [S, 6 or 24, F]: round 0 + 20 more
                if 4 * j <= Bc.shape[0] * Bc.shape[1]:
                    xs.append(pass_at(Bc, 4 * j).mean())
            curve.append(f"{np.mean(xs):.3f}" if xs else "—")
        L.append(f"| frozen: {stt} без обучения | " + " | ".join("" for _ in range(R)) + f" | {' → '.join(curve)} |")

    # ------------------------------------------------------------------ 6. bank and training
    L.append("\n## 6. Банк и обучение по раундам (seed по отдельности)\n")
    L.append("Ячейка: задач в банке (впервые в последнем раунде) / доля задач со всеми 4 слотами / шагов обучения. "
             "Когда все слоты заполнены, вес 1/|W| у режимов и у rftbal одинаков для этой задачи.\n")
    L.append("| seed, цепочка | " + " | ".join(f"M_{r}" for r in range(1, R + 1)) + " |")
    L.append("|---|" + "---|" * R)
    undertrained = []
    for s in seeds:
        for c in allc:
            cells = []
            for r in range(1, R + 1):
                td = os.path.join(run, f"s{s}", "swap", "train") if c == "swap" else os.path.join(run, f"s{s}", c, f"r{r}", "train")
                if c == "swap" and r != R or not os.path.exists(os.path.join(td, "bank.json")):
                    cells.append("—")
                    continue
                b = json.load(open(os.path.join(td, "bank.json")))
                res = json.load(open(os.path.join(td, "result.json"))) if os.path.exists(os.path.join(td, "result.json")) else {}
                short = res.get("steps_done", b["steps"]) < b["steps"]
                if short:
                    undertrained.append(f"s{s} {c} M_{r}: {res.get('steps_done')} из {b['steps']} шагов")
                cells.append(f"{b['problems']} ({b['new_last_round']}) / {b['full_share']:.2f} / "
                             f"{res.get('steps_done', '?')}{' **из ' + str(b['steps']) + '**' if short else ''}")
            L.append(f"| s{s} {c} | " + " | ".join(cells) + " |")

    if undertrained:
        L.append(f"\n**Раунды, остановленные лимитом времени (обучены меньше плана):** {'; '.join(undertrained)}")

    # ------------------------------------------------------------------ 7. modes
    L.append("\n## 7. Режимы: дополняют ли они друг друга (hv, 4 попытки на режим)\n")
    L.append("«по одной от каждого» — pass@4 из одной попытки каждого режима; «4 одного режима» — pass@4 из 4 попыток "
             "одного режима (среднее по режимам); выигрыш — их разность (у модели без режимов он 0 по построению). "
             "Исключительные — доля решённых задач, которые решил только один режим; Жаккар — среднее пересечение "
             "множеств решённых задач пар режимов; точность по режимам показывает схлопывание.\n")
    L.append("| seed, цепочка, модель | по одной от каждого | 4 одного режима | выигрыш | исключительные | Жаккар | "
             "точность по режимам |")
    L.append("|---|---|---|---|---|---|---|")
    for s in seeds:
        for c in [c for c in chains if START[c] != "sftmulti"]:
            for r in (0, R):
                e = E.get((s, c, r, "hv"))
                cm = complementarity(e) if e is not None else None
                if cm is None:
                    continue
                comps[f"modes:s{s}:{c}:M{r}"] = cm
                L.append(f"| s{s} {c} M_{r} | {cm['balanced4']:.3f} | {cm['within4']:.3f} | {cm['benefit']:+.3f} | "
                         f"{cm['exclusive']:.3f} | {cm['jaccard']:.3f} | "
                         f"{', '.join(f'{x:.2f}' for x in cm['per_mode_acc'])} |")

    L.append("\nРазнообразие ответов на hv (все задачи, фиксированный набор): разных ответов среди 4 попыток / доля "
             "попыток без \\boxed{}.\n")
    L.append("| цепочка | " + " | ".join(f"M_{r}" for r in evr) + " |")
    L.append("|---|" + "---|" * len(evr))
    for c in allc:
        L.append(f"| {c} | " + " | ".join(
            f"{mean_over(c, r, 'hv', 'distinct4'):.2f} / {mean_over(c, r, 'hv', 'noboxed'):.3f}"
            if mean_over(c, r, "hv", "distinct4") is not None else "—" for r in evr) + " |")

    # ------------------------------------------------------------------ 8. swap
    if swap:
        L.append(f"\n## 8. Банк wta1 или его использование: sftmulti, обученный на банке wta1 (hv, M_{R})\n")
        L.append("swap − rftbal > 0: на банке wta1 sftmulti учится лучше, чем на своём (число шагов зависит от "
                 "размера банка); wta1 − swap — сравнение полных методов на одном банке: в него входит и разница "
                 "стартовых адаптеров, поэтому смотрите и на изменение от M_0 в разделе 2.\n")
        L.append("| сравнение | pass@1 | pass@4 | pass@16 |")
        L.append("|---|---|---|---|")
        for X, Y in (("swap", "rftbal"), ("wta1", "swap")):
            if X not in allc or Y not in allc:
                continue
            cells = []
            for m in ("pass1", "pass4", "pass16"):
                t, per, _ = compare(X, Y, R, "hv", m, seeds)
                cells.append(cell(t) if t else "—")
                if t:
                    comps[f"swap:{X}-{Y}:{m}"] = {"mean": t[0], "lo": t[1], "hi": t[2], "per_seed": per}
            L.append(f"| {X} − {Y} | " + " | ".join(cells) + " |")

    L.append("\n## Как читать\n")
    L.append(f"- Каждый раунд модель учится заново от стартового адаптера на всём банке (STaR / ReST-EM): состояние "
             f"опыта — только банк. Шагов за раунд не больше {st.get('steps_max', '?')} и не больше "
             f"{st.get('epochs', '?')} проходов по банку; у всех методов одинаковые LR, P и правила банка (4 слота, "
             "последнее допущенное решение в слоте).\n"
             "- Открытия на пуле (раздел 5) — не доказательство обучения: часть задач решается просто от числа попыток. "
             "Обучение видно по hv (разделы 1–2): эти задачи не попадают в банк ни у кого.\n"
             "- rftbal и wta1 используют одинаковые вес задачи, банк и бюджет обучения, но стартуют с разных адаптеров "
             "фазы 1 (sftmulti — все эталоны без режимов, wta1 — один эталон и WTA): это сравнение полных методов, а "
             "разность изменений (M_R − M_0) лишь частично снимает разницу старта. rft отличается от rftbal только "
             "весом. swap — контроль переноса банка, а не чистое разделение «данные / их использование».\n"
             f"- LR: LoRA {st.get('lr', '?')}, FiLM в {st.get('film_lr_ratio', '?')} раз больше (соотношение фазы 1); "
             "разогрев 30 шагов, косинус до 10%.\n"
             "- \\boxed{} — фильтр формата, а не защита от угаданных ответов: выборка открытий для ручной проверки — "
             "audit_sample.md (без названий методов, ключ — audit_key.json).\n"
             "- Тест фазы 1 — проверка, что самообучение на трудных задачах не ломает остальное.")
    out = os.path.join(run, "report_self.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    summ = {f"s{s}:{c}:M{r}:{n}": {m: float(x.mean()) for m, x in v.items()} for (s, c, r, n), v in V.items()}
    json.dump({"summary": summ, "comparisons": comps, "acquisition": acq, "complete_seeds": full},
              open(os.path.join(run, "summary_self.json"), "w"), indent=1)
    audit(run, seeds, chains, R, P, E)
    return out


def audit(run, seeds, chains, R, P, E, per_chain=8):
    """First discoveries on hard pool problems in rounds >= 1, mixed over the chains of the first complete seed."""
    rng = random.Random(0)
    items = []
    for c in chains:
        s = next((s for s in seeds if all((s, c, r, "pool") in E for r in range(R))), None)
        if s is None:
            continue
        gd = [os.path.join(run, f"s{s}", f"start_{START[c]}", "gen")] + \
            [os.path.join(run, f"s{s}", c, f"r{r}", "gen") for r in range(1, R)]
        seen = np.zeros(len(P["pool"]), dtype=bool)
        cand = []
        for r in range(R):
            B = E[(s, c, r, "pool")]["B"]                                   # [S, n, N]
            now = B.any((0, 1))
            for j in np.flatnonzero(now & ~seen):
                p = P["pool"][j]
                if r >= 1 and p.get("stratum") in HARD_POOL:
                    cand.append((r, j))
            seen |= now
        rows = {}
        for r, j in rng.sample(cand, min(per_chain, len(cand))):
            if r not in rows:
                rows[r] = {}
                M = json.load(open(os.path.join(gd[r], "gen.json")))["modes"]
                for m in range(max(M, 1)):
                    for x in read_gz(os.path.join(gd[r], f"solve_{m}_pool.jsonl.gz")):
                        for t, ok in zip(x["texts"], x["correct"]):
                            if ok and last_boxed(t) is not None:
                                rows[r].setdefault(x["id"], t)
            p = P["pool"][j]
            items.append({"chain": c, "seed": s, "round": r, "id": p["id"], "stratum": p.get("stratum"),
                          "question": p["question"], "answer": p["answer"], "solution": rows[r].get(p["id"], "")})
    rng.shuffle(items)
    L = ["# Открытия для ручной проверки (методы скрыты)\n",
         "Для каждого: верно ли решение по существу, или ответ угадан / получен неверным путём? Ключ: audit_key.json.\n"]
    for k, it in enumerate(items):
        L += [f"## {k + 1}. {it['id']} ({it['stratum']}), правильный ответ {it['answer']}\n", it["question"], "",
              "```", it["solution"], "```", ""]
    with open(os.path.join(run, "audit_sample.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    json.dump([{k: it[k] for k in ("chain", "seed", "round", "id")} for it in items],
              open(os.path.join(run, "audit_key.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--chains", required=True)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--mark", default="")
    a = ap.parse_args()
    print(build(a.run, [int(s) for s in a.seeds.split(",")], a.chains.split(","), a.rounds, a.boot))
    if a.mark:
        os.makedirs(a.mark, exist_ok=True)
        open(os.path.join(a.mark, "DONE"), "w").close()
