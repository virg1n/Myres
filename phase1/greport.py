"""Report of GRPO with and without modes (phase1/grun.py) -> report_grpo.md, summary_grpo.json.

Primary endpoint, fixed before the run: pass@16 on hv at the final checkpoint, grpo_wta1 - grpo, paired bootstrap over
problems per seed and pooled over seeds (the same problems resampled for every seed); with it the same estimate for
the change from M_0, (final - M_0)_grpo_wta1 - (final - M_0)_grpo. The starts differ (sftmulti, wta1): the level
compares full methods, the change asks whether RL narrows less with modes. Each is called positive by the rule of
phase1_self: the pooled 95% interval above 0, every seed > 0 and the difference at least MIN_EFFECT; preliminary while
a planned seed is missing. Everything else is secondary: pass@k for k = 1..16, pass@1 / maj@4, test, the checkpoints
along training, the modes (complementarity, different answers, accuracy per mode, rewards per mode in training), strata
of hv, cut attempts, training dynamics, GRPO against the RFT chains of phase1_self from the same start.
Estimators of phase1/sreport.py: pass@k without replacement within a stratum (the 4 modes are strata, k / 4 attempts
of each), maj@4, the paired bootstrap, complementarity."""
import argparse
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase0.report import fmt  # noqa: E402
from phase1.grun import START  # noqa: E402
from phase1.sreport import HARD, MIN_EFFECT, boot, cell, complementarity, load, load_jsonl, read_gz, vectors  # noqa: E402

KS = tuple(range(1, 17))
NAMES = {"grpo": "GRPO, sftmulti", "grpo_wta1": "GRPO с режимами: wta1, 4 кода FiLM",
         "grpo_loramcl": "GRPO с режимами: LoRA-MCL, 4 LoRA", "grpo_wta1c": "grpo_wta1 с давлением к покрытию (абляция)"}
PAIRS = [("grpo_wta1", "grpo"), ("grpo_loramcl", "grpo"), ("grpo_wta1", "grpo_loramcl"), ("grpo_wta1c", "grpo_wta1")]
RFT = {"grpo": ("rftbal", "rft"), "grpo_wta1": ("wta1",), "grpo_loramcl": ("loramcl",), "grpo_wta1c": ("wta1",)}
PLANNED_SEEDS = (0, 1)       # the registered design: a verdict without one of them is preliminary
LONG_SLACK = 4               # re-tokenized text this close to max_tokens: the attempt ran into the limit


def mname(m):
    return m.replace("pass", "pass@").replace("maj4", "maj@4")


def long_share(gdir, name, tok, max_tokens):
    """Share of the attempts whose text re-tokenizes to >= max_tokens - LONG_SLACK tokens (approximately: cut)."""
    texts = []
    M = json.load(open(os.path.join(gdir, "gen.json")))["modes"]
    for m in range(max(M, 1)):
        for r in read_gz(os.path.join(gdir, f"solve_{m}_{name}.jsonl.gz")):
            texts += r["texts"]
    n = [len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]] if texts else []
    return float(np.mean([k >= max_tokens - LONG_SLACK for k in n])) if n else None


def window(recs, key, part):
    """Mean of a per-step record over the first (part < 0) or last (part > 0) 10% of the steps."""
    if not recs:
        return None
    w = max(1, len(recs) // 10)
    xs = [r[key] for r in (recs[:w] if part < 0 else recs[-w:]) if r.get(key) is not None]
    return np.mean(xs, axis=0) if xs else None


def build(run, data, m0root, self_run, seeds, chains, n, boot_n=1000):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    P = {k: load_jsonl(os.path.join(data, f"{k}.jsonl")) for k in ("hv", "test")}
    st = json.load(open(os.path.join(run, "state.json"))) if os.path.exists(os.path.join(run, "state.json")) else {}
    rng = np.random.default_rng(0)

    def gdir(s, c, j):
        return os.path.join(m0root, f"s{s}", f"start_{START[c]}", "gen") if j == 0 else \
            os.path.join(run, f"s{s}", c, f"c{j}", "gen")

    V, E, rejected, cache = {}, {}, [], {}
    for s in seeds:
        for c in chains:
            for j in range(n + 1):
                for name in ("hv", "test"):
                    k = (gdir(s, c, j), name)
                    if k not in cache:
                        try:
                            e = load(k[0], name, P[name])
                        except (ValueError, OSError) as ex:
                            rejected.append(str(ex))
                            e = None
                        cache[k] = (e, vectors(e, np.random.default_rng(0), KS) if e is not None else None)
                    if cache[k][0] is not None:
                        E[(s, c, j, name)], V[(s, c, j, name)] = cache[k]
    res, recs = {}, {}
    for s in seeds:
        for c in chains:
            td = os.path.join(run, f"s{s}", c, "train")
            if os.path.exists(os.path.join(td, "result.json")):
                res[(s, c)] = json.load(open(os.path.join(td, "result.json")))
            if os.path.exists(os.path.join(td, "steps.jsonl")):
                recs[(s, c)] = load_jsonl(os.path.join(td, "steps.jsonl"))
    done = {s: [c for c in chains if (s, c, n, "hv") in V and (s, c, 0, "hv") in V] for s in seeds}
    cohort = {c: [s for s in seeds if c in done[s]] for c in chains}
    full = [s for s in seeds if all(c in done[s] for c in chains)]
    under = sorted((s, c) for (s, c), r in res.items() if not r["complete"])
    short = [f"s{s} {c}: {res[(s, c)]['steps_done']} из {res[(s, c)]['steps']} шагов" for s, c in under]
    idx = {"hv": rng.integers(0, len(P["hv"]), (boot_n, len(P["hv"]))),
           "test": rng.integers(0, len(P["test"]), (boot_n, len(P["test"])))}
    comps = {}

    def diff(X, Y, name, m, change=False, j=n):
        """(pooled (mean, lo, hi), per seed (mean, lo, hi), seeds): X - Y at checkpoint j, or of the changes from M_0."""
        js = (0, j) if change else (j,)
        ss = [s for s in seeds if all((s, c, q, name) in V and m in V[(s, c, q, name)] for c in (X, Y) for q in js)]
        if not ss:
            return None, [], []
        d = lambda s, c: V[(s, c, j, name)][m] - (V[(s, c, 0, name)][m] if change else 0)
        dv = np.stack([d(s, X) - d(s, Y) for s in ss])
        return boot(dv, idx[name]), [boot(x, idx[name]) for x in dv], ss

    def mean_over(c, j, name, m, ss=None):
        xs = [V[(s, c, j, name)][m].mean() for s in (cohort[c] if ss is None else ss)
              if (s, c, j, name) in V and m in V[(s, c, j, name)]]
        return float(np.mean(xs)) if xs else None

    S = next((r["steps"] for r in res.values()), None)
    r0 = next(iter(res.values()), {})
    L = ["# GRPO с режимами и без: сохраняет ли RL с режимами покрытие\n"]
    L.append(f"- seeds: {seeds} (все цепочки дошли до финала: {full or 'нет'}); цепочки: {chains}; hv {len(P['hv'])} "
             f"задач, тест {len(P['test'])}.")
    if r0:
        L.append(f"- обучение: {S} шагов × {r0['P']} задач × {r0['G']} роллаутов = {S * r0['P'] * r0['G']} роллаутов на "
                 f"цепочку; пул {r0['pool']} задач (выброшено лёгких {r0['dropped_easy']}); LR {st.get('lr', '?')}, у FiLM "
                 f"в {st.get('film_lr_ratio', '?')} раз больше; T роллаутов {st.get('temp', '?')}, max_tokens "
                 f"{st.get('max_tokens', '?')}; чекпоинты на шагах {r0['ckpts']}.")
    sf = os.path.join(run, "steps.json")
    if os.path.exists(sf):
        x = json.load(open(sf))
        L.append(f"- шаги по калибровке: {x['steps']} (самый медленный вид старта за {x['train_hours']} ч; мин/шаг: "
                 + ", ".join(f"{os.path.basename(os.path.dirname(k))} {v:.2f}" for k, v in x["min_per_step"].items())
                 + "); на деле: " + ", ".join(f"s{s} {c} {r['min_per_step']:.2f}" for (s, c), r in sorted(res.items())) + ".")
    bad = {j: x["status"] for j, x in st.get("jobs", {}).items() if x.get("status") in ("failed", "skipped")}
    if bad:
        L.append(f"- **не выполнены:** {bad} (см. logs/)")
    missing = [f"s{s}:{c}" for s in seeds for c in chains if c not in done[s]]
    if missing:
        L.append(f"- **нет финального чекпоинта или M_0:** {', '.join(missing)}")
    if short:
        L.append(f"- **недоучены (остановлены лимитом времени):** {'; '.join(short)}")
    if rejected:
        L.append(f"- **отклонены файлы:** {rejected}")

    # ------------------------------------------------------------------ 1. primary
    L.append(f"\n## 1. Главная метрика (зафиксирована до запуска): pass@16 на hv у финального чекпоинта, grpo_wta1 − grpo\n")
    L.append(f"Уровень — разность финальных моделей (в неё входит и разница стартов sftmulti / wta1); изменение — "
             f"разность (финал − M_0) двух цепочек: сужает ли RL меньше с режимами. Вывод «лучше» для каждой строки: общий "
             f"по seed 95% интервал выше 0, у каждого seed разность > 0 и общая разность ≥ {MIN_EFFECT}; пока нет "
             f"какого-то из запланированных seed {list(PLANNED_SEEDS)}, вывод предварительный. Интервал — "
             f"бутстрап по задачам при этих обучениях, разброс между обучениями виден только по столбцу seed. Строки "
             f"после первых двух — вторичные.\n")
    L.append("| сравнение | разность [95%] по всем seed | по seed [95%] | вывод |")
    L.append("|---|---|---|---|")
    for X, Y in [p for p in PAIRS if p[0] in chains and p[1] in chains]:
        for change in (False, True):
            t, per, ss = diff(X, Y, "hv", "pass16", change)
            what = f"{X} − {Y}, {'изменение от M_0' if change else 'уровень'}" + \
                (" (главное)" if (X, Y) == ("grpo_wta1", "grpo") else "")
            if t is None:
                L.append(f"| {what} | — | — | нет данных |")
                continue
            win = t[1] > 0 and all(p[0] > 0 for p in per) and t[0] >= MIN_EFFECT
            lack = [s for s in sorted(set(seeds) | set(PLANNED_SEEDS)) if s not in ss]
            und = [x for x in under if x[1] in (X, Y)]
            verdict = (f"**{X} лучше**" if win else "не показано") + (f" (предварительно: нет seed {lack})" if lack else "") \
                + (" (есть недоученные)" if und else "")
            comps[f"primary:{X}-{Y}:{'change' if change else 'level'}"] = {
                "mean": t[0], "lo": t[1], "hi": t[2], "per_seed": dict(zip(ss, per)), "missing_seeds": lack,
                "win": bool(win and not lack and not und)}
            L.append(f"| {what} | {cell(t)} | {'; '.join(f's{x}: {cell(p)}' for x, p in zip(ss, per))} | {verdict} |")
    t, _, _ = diff("grpo_wta1", "grpo", "hv", "pass16", j=0) if {"grpo_wta1", "grpo"} <= set(chains) else (None, 0, 0)
    if t is not None:
        L.append(f"\nРазница стартов (M_0, те же генерации, что в phase1_self): grpo_wta1 − grpo = {cell(t)}.")

    # ------------------------------------------------------------------ 2. pass@k
    L.append(f"\n## 2. pass@k на hv, k = 1…16: сужает ли RL\n")
    L.append("Несмещённая оценка без возвращения; у моделей с режимами k попыток поровну по режимам (для k, не кратного 4, "
             "лишние попытки — у случайных режимов). Среднее по seed, где цепочка дошла до финала (в скобках); Δ = финал − "
             "M_0 внутри seed. Сужение — Δ > 0 при малых k и Δ < 0 при больших.\n")
    L.append("| цепочка | модель | " + " | ".join(f"k={k}" for k in KS) + " |")
    L.append("|---|---|" + "---|" * len(KS))
    for c in chains:
        if not cohort[c]:
            continue
        a0 = [mean_over(c, 0, "hv", f"pass{k}") for k in KS]
        a1 = [mean_over(c, n, "hv", f"pass{k}") for k in KS]
        L.append(f"| {c} (seed {cohort[c]}) | M_0 | " + " | ".join(fmt(x) for x in a0) + " |")
        L.append(f"| {c} | финал | " + " | ".join(fmt(x) for x in a1) + " |")
        dl = [x1 - x0 if x0 is not None and x1 is not None else None for x0, x1 in zip(a0, a1)]
        L.append(f"| {c} | Δ | " + " | ".join(f"{x:+.3f}" if x is not None else "—" for x in dl) + " |")
        cross = next((k for k, x in zip(KS, dl) if x is not None and x <= 0), None)
        comps[f"passk:{c}"] = {"M0": a0, "final": a1, "delta": dl, "first_k_not_better": cross}
    L.append("\nРазность изменений по k (финал − M_0)_X − (финал − M_0)_Y:\n")
    ks = (1, 2, 4, 8, 16)
    L.append("| сравнение | " + " | ".join(f"k={k}" for k in ks) + " |")
    L.append("|---|" + "---|" * len(ks))
    for X, Y in [p for p in PAIRS if p[0] in chains and p[1] in chains]:
        cells = []
        for k in ks:
            t, _, _ = diff(X, Y, "hv", f"pass{k}", True)
            cells.append(cell(t) if t else "—")
            if t:
                comps[f"change:{X}-{Y}:pass{k}"] = {"mean": t[0], "lo": t[1], "hi": t[2]}
        L.append(f"| {X} − {Y} | " + " | ".join(cells) + " |")

    # ------------------------------------------------------------------ 3. along training
    L.append("\n## 3. По ходу обучения (hv, 16 попыток на каждом чекпоинте)\n")
    L.append("Ячейка: pass@1 / pass@4 / pass@16, среднее по тем же seed, что в разделе 2.\n")
    steps = r0.get("ckpts", list(range(1, n + 1)))
    L.append("| цепочка | M_0 | " + " | ".join(f"шаг {x}" for x in steps) + " |")
    L.append("|---|---|" + "---|" * n)
    for c in chains:
        if cohort[c]:
            L.append(f"| {c} | " + " | ".join(" / ".join(fmt(mean_over(c, j, "hv", m)) for m in ("pass1", "pass4", "pass16"))
                                           for j in range(n + 1)) + " |")

    # ------------------------------------------------------------------ 4. pass@1, maj@4, test
    L.append("\n## 4. pass@1, maj@4 и тест фазы 1 (4 попытки на задачу)\n")
    cols = [("hv", "pass1"), ("hv", "maj4"), ("test", "pass1"), ("test", "pass4"), ("test", "maj4")]
    L.append("| цепочка | " + " | ".join(f"{nm} {mname(m)} M_0 → финал" for nm, m in cols) + " |")
    L.append("|---|" + "---|" * len(cols))
    for c in chains:
        if cohort[c]:
            L.append(f"| {c} | " + " | ".join(f"{fmt(mean_over(c, 0, nm, m))} → {fmt(mean_over(c, n, nm, m))}"
                                             for nm, m in cols) + " |")
    L.append("\nПарные разности у финала (уровень / изменение от M_0):\n")
    L.append("| метрика | " + " | ".join(f"{X} − {Y}" for X, Y in PAIRS if X in chains and Y in chains) + " |")
    L.append("|---|" + "---|" * sum(X in chains and Y in chains for X, Y in PAIRS))
    for nm, m in [("hv", "pass1"), ("hv", "pass4"), ("hv", "maj4")] + cols[2:]:
        cells = []
        for X, Y in [p for p in PAIRS if p[0] in chains and p[1] in chains]:
            t1, per, _ = diff(X, Y, nm, m)
            t2, _, _ = diff(X, Y, nm, m, True)
            if t1:
                comps[f"{nm}:{m}:{X}-{Y}"] = {"mean": t1[0], "lo": t1[1], "hi": t1[2], "per_seed": [p[0] for p in per]}
            if t2:
                comps[f"{nm}:{m}:{X}-{Y}:change"] = {"mean": t2[0], "lo": t2[1], "hi": t2[2]}
            cells.append(f"{cell(t1) if t1 else '—'} / {cell(t2) if t2 else '—'}")
        L.append(f"| {nm} {mname(m)} | " + " | ".join(cells) + " |")

    # ------------------------------------------------------------------ 5. modes
    L.append("\n## 5. Режимы: не схлопнул ли их GRPO (hv, 4 попытки на режим)\n")
    L.append("«по одной от каждого» — pass@4 из одной попытки каждого режима; «4 одного режима» — pass@4 из 4 попыток "
             "одного режима (среднее по режимам); выигрыш — их разность (0, если режимы одинаковы). Исключительные — доля "
             "решённых задач, решённых только одним режимом; Жаккар — пересечение множеств решённых задач пар режимов. "
             "При 4 попытках на режим и одна политика даёт выигрыш > 0, исключительные > 0 и Жаккар < 1 (шум "
             "сэмплирования), поэтому нулевая линия — строки «псевдорежимы»: 16 сэмплов модели без режимов, разбитые на 4 "
             "группы по 4. Схлопывание — режимы приближаются к нулевой линии (при похожей точности); одинаковая средняя "
             "точность режимов сама по себе о схлопывании не говорит.\n")
    L.append("| seed, цепочка, модель | по одной от каждого | 4 одного режима | выигрыш | исключительные | Жаккар | "
             "точность по режимам |")
    L.append("|---|---|---|---|---|---|---|")
    for s in seeds:
        for c in chains:
            for j in range(n + 1):
                e = E.get((s, c, j, "hv"))
                if e is not None and START[c] == "sftmulti":         # one policy split into 4 pseudo-modes
                    C = e["C"]
                    e = {"C": C[0].reshape(4, -1, C.shape[2])} if C.shape[0] == 1 and C.shape[1] % 4 == 0 else None
                cm = complementarity(e) if e is not None else None
                if cm is None:
                    continue
                comps[f"modes:s{s}:{c}:c{j}"] = cm
                L.append(f"| s{s} {c}{' (псевдорежимы)' if START[c] == 'sftmulti' else ''} "
                         f"{'M_0' if j == 0 else f'шаг {steps[j - 1]}'} | {cm['balanced4']:.3f} | "
                         f"{cm['within4']:.3f} | {cm['benefit']:+.3f} | {cm['exclusive']:.3f} | {cm['jaccard']:.3f} | "
                         f"{', '.join(f'{x:.2f}' for x in cm['per_mode_acc'])} |")
    L.append("\nРазные ответы: среднее число разных ответов среди первых попыток 4 режимов (у модели без режимов — среди 4 "
             "сэмплов) на hv.\n")
    L.append("| цепочка | M_0 | " + " | ".join(f"шаг {x}" for x in steps) + " |")
    L.append("|---|---|" + "---|" * n)
    for c in chains:
        if cohort[c]:
            L.append(f"| {c} | " + " | ".join(fmt(mean_over(c, j, "hv", "distinct4"), 2) for j in range(n + 1)) + " |")
    mode_recs = [(s, c) for (s, c) in sorted(recs) if START[c] != "sftmulti"]
    if mode_recs:
        L.append("\nДоля верных роллаутов по режимам при обучении (первые / последние 10% шагов). Близкие награды не "
                 "значат, что режимы решают одно и то же: это видно только по таблице выше.\n")
        L.append("| seed, цепочка | первые 10% | последние 10% |")
        L.append("|---|---|---|")
        for s, c in mode_recs:
            f, l = window(recs[(s, c)], "reward_mode", -1), window(recs[(s, c)], "reward_mode", 1)
            L.append(f"| s{s} {c} | {', '.join(f'{x:.2f}' for x in f) if f is not None else '—'} | "
                     f"{', '.join(f'{x:.2f}' for x in l) if l is not None else '—'} |")

    # ------------------------------------------------------------------ 6. strata
    L.append("\n## 6. Страты hv: pass@16, M_0 → финал (Δ)\n")
    L.append("| цепочка | " + " | ".join(HARD) + " |")
    L.append("|---|" + "---|" * len(HARD))
    strat = np.array([p.get("stratum") for p in P["hv"]])
    for c in chains:
        row = []
        for h in HARD:
            msk = strat == h
            x0 = [V[(s, c, 0, "hv")]["pass16"][msk].mean() for s in cohort[c] if msk.any()]
            x1 = [V[(s, c, n, "hv")]["pass16"][msk].mean() for s in cohort[c] if msk.any()]
            row.append(f"{np.mean(x0):.3f} → {np.mean(x1):.3f} ({np.mean(x1) - np.mean(x0):+.3f}; n={int(msk.sum())})"
                       if x0 else "—")
        L.append(f"| {c} | " + " | ".join(row) + " |")

    # ------------------------------------------------------------------ 7. cut attempts
    L.append("\n## 7. Обрезанные попытки\n")
    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(st["model_path"])
    except Exception as ex:          # no model directory here: only the exact columns
        L.append(f"(токенизатор не загрузился: {type(ex).__name__}; столбец «≈ max_tokens» пропущен)\n")
    mt = st.get("max_tokens", 1200)
    L.append(f"hv: доля попыток без \\boxed{{}} (точно) и упёршихся в max_tokens = {mt} (≈: текст заново разбит на токены, "
             f"≥ {mt - LONG_SLACK}). Обучение: доля роллаутов без стоп-токена (точно) и без \\boxed{{}}, первые / последние "
             f"10% шагов (T роллаутов {st.get('temp', '?')}, у оценок 0.7).\n")
    L.append("| цепочка | hv без \\boxed, M_0 → финал | hv ≈ max_tokens, M_0 → финал | обучение: без стоп-токена | "
             "обучение: без \\boxed |")
    L.append("|---|---|---|---|---|")
    lcache = {}

    def long_mean(c, j):
        xs = []
        for s in cohort[c]:
            g = gdir(s, c, j)
            if g not in lcache:
                lcache[g] = long_share(g, "hv", tok, mt)
            xs.append(lcache[g])
        return float(np.mean(xs)) if xs and None not in xs else None

    for c in chains:
        if not cohort[c]:
            continue
        lg = f"{fmt(long_mean(c, 0))} → {fmt(long_mean(c, n))}" if tok is not None else "—"
        tr = [(window(recs[(s, c)], "trunc", -1), window(recs[(s, c)], "trunc", 1),
               window(recs[(s, c)], "noboxed", -1), window(recs[(s, c)], "noboxed", 1))
              for s in cohort[c] if recs.get((s, c))]
        tm = [float(np.mean([x[i] for x in tr])) if tr else None for i in range(4)]
        if tok is not None:
            comps[f"long:{c}"] = {"M0": long_mean(c, 0), "final": long_mean(c, n)}
        L.append(f"| {c} | {fmt(mean_over(c, 0, 'hv', 'noboxed'))} → {fmt(mean_over(c, n, 'hv', 'noboxed'))} | {lg} | "
                 f"{fmt(tm[0])} → {fmt(tm[1])} | {fmt(tm[2])} → {fmt(tm[3])} |")

    # ------------------------------------------------------------------ 8. training
    L.append("\n## 8. Обучение (seed по отдельности)\n")
    L.append("Первые / последние 10% шагов: доля верных роллаутов, доля задач с разными наградами в группе (только они дают "
             "градиент), длина роллаута в токенах, NLL сэмплированных токенов на роллаутах с градиентом (не оценка энтропии "
             "политики: какие роллауты в неё входят, зависит от наград).\n")
    L.append("| seed, цепочка | шагов | часов (мин/шаг) | награда | группы с градиентом | длина | NLL |")
    L.append("|---|---|---|---|---|---|---|")
    for (s, c), r in sorted(res.items()):
        rc = recs.get((s, c), [])
        pair = lambda k, d=3: f"{fmt(window(rc, k, -1), d)} → {fmt(window(rc, k, 1), d)}"
        L.append(f"| s{s} {c} | {r['steps_done']}{'' if r['complete'] else ' **из ' + str(r['steps']) + '**'} | "
                 f"{r['hours']:.2f} ({r['min_per_step']:.2f}) | {pair('reward')} | {pair('mixed', 2)} | {pair('len', 0)} | "
                 f"{pair('nll')} |")

    # ------------------------------------------------------------------ 9. GRPO vs RFT
    L.append("\n## 9. GRPO против RFT того же старта (phase1_self, hv, финал против M_R)\n")
    sj = os.path.join(self_run, "state.json") if self_run else ""
    if not self_run or not os.path.exists(sj):
        L.append(f"Раздел пропущен: нет phase1_self ({sj or 'не задан'}).")
    else:
        R = json.load(open(sj))["rounds"]
        meta = os.path.join(self_run, "data", "meta.json")
        npool = json.load(open(meta))["sizes"]["pool"] if os.path.exists(meta) else None
        if not os.path.exists(os.path.join(self_run, "report_self.md")):
            L.append("(отчёта phase1_self нет: цепочки RFT могли не дойти до конца)\n")
        L.append(f"Старт и M_0 у пары общие, так что разность финалов — это и разность изменений. Бюджет разный: RFT — "
                 f"{R} раундов по 4 попытки на {npool or '?'} задач пула (≈ {R * 4 * npool if npool else '?'} решений) "
                 f"плюс обучение на банке; GRPO — {S * r0['P'] * r0['G'] if r0 else '?'} роллаутов.\n")
        L.append("| GRPO − RFT | pass@1 | pass@4 | pass@16 | seed |")
        L.append("|---|---|---|---|---|")
        skipped = []
        for c in chains:
            for Y in RFT[c]:
                for s in seeds:
                    g = os.path.join(self_run, f"s{s}", Y, f"r{R}", "gen")
                    if (s, f"rft:{Y}", n, "hv") in V:
                        continue
                    try:
                        e = load(g, "hv", P["hv"])
                    except (ValueError, OSError) as ex:
                        e = None
                        skipped.append(f"{Y} s{s}: {ex}")
                    if e is not None:
                        V[(s, f"rft:{Y}", n, "hv")] = vectors(e, np.random.default_rng(0), KS)
                    elif not os.path.exists(os.path.join(g, "DONE")):
                        skipped.append(f"{Y} s{s}: нет {g}")
                cells, used = [], []
                for m in ("pass1", "pass4", "pass16"):
                    t, per, ss = diff(c, f"rft:{Y}", "hv", m)
                    cells.append(cell(t) if t else "—")
                    used = ss or used
                    if t:
                        comps[f"rft:{c}-{Y}:{m}"] = {"mean": t[0], "lo": t[1], "hi": t[2], "per_seed": [p[0] for p in per]}
                L.append(f"| {c} − {Y} | " + " | ".join(cells) + f" | {used or '—'} |")
        if skipped:
            L.append(f"\nНет данных: {'; '.join(sorted(set(skipped)))}")

    L.append("\n## Как читать\n")
    L.append("- Обе цепочки seed видят одни и те же задачи в одном порядке, одинаковые G, P, шаги, LR, loss, max_tokens и "
             "сэмплирование роллаутов. Отличаются только режимы: у режимов 16 роллаутов задачи — по 4 на код FiLM (или "
             "адаптер), группа и преимущество те же, роллаут учит только свой режим (и общую LoRA).\n"
             "- Преимущество Dr. GRPO: r − среднее группы, без деления на SD и без нормировки на длину; один шаг оптимизатора "
             "на батч роллаутов (on-policy, клиппинг не нужен), без KL. Группа с одинаковыми наградами градиента не даёт.\n"
             "- Роллауты при обучении — T = 1 без top-p / top-k (HF generate в процессе тренера); оценки — vLLM с "
             "сэмплированием phase0.gen (T = 0.7, top-p 0.8, top-k 20), те же, что у M_0 и цепочек RFT в phase1_self.\n"
             "- Разность уровней включает разницу стартов; разность изменений от M_0 снимает её лишь частично (от разного "
             "старта RL может двигаться по-разному).\n"
             "- В режимах нет явного давления к разнообразию: в ожидании каждый режим максимизирует свой pass@1, как и "
             "обычный GRPO. Схлопывание режимов видно в разделе 5.\n"
             "- Сравнение с RFT (раздел 9) — разные бюджеты и разные алгоритмы; это ориентир, а не контролируемый опыт.")
    out = os.path.join(run, "report_grpo.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    summ = {f"s{s}:{c}:c{j}:{nm}": {m: float(x.mean()) for m, x in v.items()} for (s, c, j, nm), v in V.items()}
    json.dump({"summary": summ, "comparisons": comps, "complete_seeds": full,
               "train": {f"s{s}:{c}": r for (s, c), r in res.items()}},
              open(os.path.join(run, "summary_grpo.json"), "w"), indent=1, default=float)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True, help="dir with hv.jsonl / test.jsonl")
    ap.add_argument("--m0", required=True, help="root of the M_0 generations: <m0>/s<seed>/start_<start>/gen")
    ap.add_argument("--self_run", default="", help="phase1_self run for the comparison with RFT (empty: skipped)")
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--chains", required=True)
    ap.add_argument("--evals", type=int, required=True)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--mark", default="")
    a = ap.parse_args()
    print(build(a.run, a.data, a.m0, a.self_run, [int(s) for s in a.seeds.split(",")], a.chains.split(","), a.evals,
                a.boot))
    if a.mark:
        os.makedirs(a.mark, exist_ok=True)
        open(os.path.join(a.mark, "DONE"), "w").close()
