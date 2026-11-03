"""Report of the equal-parameter run (phase1/rank_run.py): the new ranks against the r = 32 runs of the same seeds.

Pairing and bootstrap as in phase1/seeds_report.py: the same problem indices for all methods, one replicate of a
comparison is the mean over seeds of X(seed) - R(seed)."""
import argparse
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase0.report import fmt, load_methods  # noqa: E402
from phase1.metrics import load_eval  # noqa: E402
from phase1.seeds_report import METRICS, OLD, P0, cell, compare, eval_dir, metric, vectors, verifier_auroc  # noqa: E402

NEW = [("wta1_r128", "wta1, LoRA r = 128 + FiLM"), ("loramcl_r10", "LoRA-MCL, 4 адаптера r = 10")]
REF = [("wta1", "wta1, LoRA r = 32 + FiLM"), ("loramcl", "LoRA-MCL, 4 адаптера r = 32"),
       ("sftmulti", "SFT-multi, LoRA r = 32")]
# (X, R, what the pair answers)
PAIRS = [("wta1_r128", "loramcl", "равные параметры: общая LoRA + FiLM против 4 отдельных LoRA"),
         ("wta1", "loramcl_r10", "равные параметры, то же при ~43M"),
         ("wta1", "loramcl", "исходная пара r = 32 на тех же seed, параметры не равны"),
         ("wta1_r128", "wta1", "что дал ранг wta1"),
         ("loramcl_r10", "loramcl", "что дал ранг LoRA-MCL"),
         ("wta1_r128", "sftmulti", "против SFT"), ("loramcl_r10", "sftmulti", "против SFT"),
         ("wta1_r128", P0, "против промптов-стратегий"), ("loramcl_r10", P0, "против промптов-стратегий")]


def where(run, seeds_run, phase1, v, s, kind):
    """Directory of the evaluation (kind "eval") or the result.json of the training (kind "train")."""
    if v in dict(NEW):
        d = os.path.join(run, kind, f"{v}_s{s}")
    elif kind == "eval":
        return eval_dir(seeds_run, phase1, v, s)
    else:
        d = os.path.join(phase1, "train", v) if phase1 and s == 0 and v in OLD else \
            os.path.join(seeds_run, "train", f"{v}_s{s}")
    return d if kind == "eval" else os.path.join(d, "result.json")


def load(run, seeds_run, phase1, phase0, test, seeds):
    """methods[name][seed] = A; the phase-0 strategies have the single seed None."""
    methods, missing = {}, []
    if phase0 and os.path.isdir(phase0):
        try:
            tests, m0 = load_methods(phase0)
            if "strategies" in m0 and [t["id"] for t in tests] == [t["id"] for t in test]:
                methods[P0] = {None: m0["strategies"][0]}
            else:
                missing.append("стратегии фазы 0")
        except ValueError as e:
            missing.append(f"стратегии фазы 0 (отклонены: {e})")
    for v, _ in NEW + REF:
        for s in seeds:
            try:
                r = load_eval(where(run, seeds_run, phase1, v, s, "eval"), test, "test")
            except ValueError as e:
                missing.append(f"{v} seed {s} (отклонён: {e})")
                continue
            if r is None:
                missing.append(f"{v} seed {s}")
            else:
                methods.setdefault(v, {})[s] = r[0]
    return methods, missing


def build(run, data, seeds_run, phase1, phase0, seeds, boot=1000):
    warnings.filterwarnings("ignore", category=RuntimeWarning)     # means over bootstrap samples without errors
    test = [json.loads(l) for l in open(os.path.join(data, "test.jsonl"))]
    N = len(test)
    methods, missing = load(run, seeds_run, phase1, phase0, test, seeds)
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

    L = ["# Фаза 1: равное число обучаемых параметров\n"]
    L.append(f"- задач в тесте: {N}; seed: {', '.join(map(str, seeds))}. Варианты r = 32 тех же seed взяты из "
             f"phase1_seeds без перезапуска (wta1 и sftmulti seed 0 — из фазы 1).")
    st = json.load(open(os.path.join(run, "state.json"))) if os.path.exists(os.path.join(run, "state.json")) else {}
    if st.get("alphas"):
        L.append(f"- alpha LoRA ({st.get('alpha_rule')}): {st['alphas']}; у вариантов r = 32 alpha = 64.")
    if missing:
        L.append(f"- **нет данных:** {', '.join(missing)}")
    bad = {k: v["status"] for k, v in st.get("jobs", {}).items() if v.get("status") in ("failed", "skipped")}
    if bad:
        L.append(f"- **не выполнены:** {bad} (см. logs/)")

    L.append("\n## 1. Среднее ± SD по seed\n")
    L.append("| метод | seed | " + " | ".join(n for _, n in METRICS) + " | AUROC Yes/No: свой / чужой |")
    L.append("|---|---|" + "---|" * (len(METRICS) + 1))
    summary = {}
    for name, desc in NEW + REF + [(P0, "стратегии через промпт, фаза 0 (один прогон)")]:
        if name not in methods:
            continue
        ss = sorted(methods[name], key=str)
        row, summ = [], {}
        for k, _ in METRICS:
            x = np.array([point[name, s][k] for s in ss])
            m = float(np.nanmean(x))
            sd = float(np.nanstd(x, ddof=1)) if np.isfinite(x).sum() > 1 else float("nan")
            summ[k] = {"mean": m, "sd": sd, "per_seed": {str(s): float(v) for s, v in zip(ss, x)}}
            row.append(fmt(m) + (f" ± {sd:.3f}" if np.isfinite(sd) else ""))
        va = np.array([vers[name, s] for s in ss])
        av = [float(np.nanmean(va[:, i])) for i in (0, 1)]
        summ["auroc_own"], summ["auroc_other"] = av
        summary[name] = summ
        L.append(f"| {name} ({desc}) | {', '.join(str(s) for s in ss) if ss != [None] else '—'} | "
                 + " | ".join(row) + f" | {fmt(av[0])} / {fmt(av[1])} |")

    L.append("\n## 2. Парные сравнения, X − R\n")
    L.append("Ячейка: среднее по seed ± SD разностей по seed [95% бутстрап по задачам] и число seed, где знак "
             "разности совпал со средним. Для pass@1, maj@4, pass@4 лучше больше, для φ и одинакового неверного "
             "ответа — меньше.\n")
    L.append("| X − R | вопрос | " + " | ".join(n for _, n in METRICS) + " |")
    L.append("|---|---|" + "---|" * len(METRICS))
    comps = {}
    for X, R, what in PAIRS:
        if X not in methods or R not in methods:
            continue
        pairs = [(s, None) for s in sorted(methods[X])] if R == P0 else \
            [(s, s) for s in sorted(methods[X]) if s in methods[R]]
        if not pairs:
            continue
        cells = []
        for k, _ in METRICS:
            d = compare(point, reps, X, R, pairs, k)
            comps[f"{X}-{R}:{k}"] = d
            cells.append(cell(d))
        L.append(f"| {X} − {R} | {what} | " + " | ".join(cells) + " |")

    L.append("\n## 3. Обучение\n")
    L.append("| вариант | seed | r | alpha | параметров | шагов | эпох | токенов | часов | доля решений по режимам |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for v, _ in NEW + REF:
        for s in seeds:
            f = where(run, seeds_run, phase1, v, s, "train")
            if not os.path.exists(f):
                continue
            r = json.load(open(f))
            share = r.get("mode_share")
            L.append(f"| {v} | {s} | {r.get('lora_r', 32)} | {r.get('lora_alpha', 64):g} | "
                     f"{str(round(r['trainable_M'], 1)) + 'M' if r.get('trainable_M') else '—'} | "
                     f"{r.get('steps_done')} | {r.get('epochs_done', 0):.2f} | {r['tokens'] / 1e6:.1f}M | "
                     f"{r['hours']:.2f} | {[round(x, 2) for x in share] if share else '—'} |")
    alpha_note = ("- alpha: масштаб alpha / r перенесён с r = 32 по правилу rsLoRA (∝ 1 / √r): при r = 128 он 1, "
                  "при r = 10 — 3.58. Это эвристика из анализа rsLoRA, а не гарантия одинакового шага при AdamW, "
                  "обрезке градиента и FiLM; LR одинаковый у всех.\n"
                  if st.get("alpha_rule", "rs") == "rs" else
                  "- alpha = 2r, как в фазе 1: масштаб 2 при любом ранге, поэтому при том же LR обновление r = 128 "
                  "больше, а r = 10 меньше, чем у r = 32.\n")
    L.append("\n## Как читать\n")
    L.append("- Параметры выровнены, токены — нет: обучение ограничено 2 часами, а в LoRA-MCL каждый пример проходит "
             "через все 4 адаптера, поэтому он видит примерно вдвое меньше токенов (раздел 3).\n"
             + alpha_note +
             "- Seed всего два: SD по двум точкам грубое. Бутстрап-интервал учитывает только случайность задач при "
             "этих обученных моделях, а не разброс между обучениями, поэтому узкий интервал при двух seed ещё не "
             "означает воспроизводимого преимущества.\n"
             "- Главное — сравнить строку «wta1 − loramcl» (исходная пара) с двумя строками при примерно равных "
             "параметрах (147.0M против 139.5M, 42.5M против 43.6M) и равном времени. Если разность меняется, это "
             "согласуется с ролью числа параметров, но ранг меняет и масштаб, и скорость (число шагов за 2 часа, "
             "раздел 3), так что одним числом параметров это не доказано.")
    out = os.path.join(run, "report_rank.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    json.dump({"summary": summary, "comparisons": comps, "missing": missing},
              open(os.path.join(run, "summary_rank.json"), "w"), default=float, indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--seeds_run", required=True)
    ap.add_argument("--phase1", default="")
    ap.add_argument("--phase0", default="")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--mark", default="")
    a = ap.parse_args()
    print(build(a.run, a.data, a.seeds_run, a.phase1, a.phase0, [int(s) for s in a.seeds.split(",")], a.boot))
    if a.mark:
        os.makedirs(a.mark, exist_ok=True)
        open(os.path.join(a.mark, "DONE"), "w").close()
