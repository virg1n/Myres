"""Do the 4 prompt strategies of phase 0 solve (and fail) the same problems?  CPU only, reads phase-0 outputs.

Why: phase-1 modes learned the strategies (match / oracle), yet pass@16 did not move. If the strategies themselves
fail together, no split of the modes by strategy can raise pass@k. This measures that ceiling on the base model.

Sets (every problem with >= 2 samples of each of the 4 strategies):
  test  phase-0 test (unselected; also 8 neutral samples, the "temperature only" baseline)
  val   phase-1 val (problems with 10-90% base success over all attempts: selected)
  pool  every pool problem (phase-0 pool + phase-1 hard pool)
All estimators are unbiased per problem (the samples of a strategy are i.i.d.); 95% bootstrap over problems.

  pass@2 same vs different   two samples of one strategy vs one sample each of two strategies: the gain is what
                             switching strategy buys at the same budget
  pass@4 / pass@8            test: 1 or 2 samples of each strategy vs 4 / 8 neutral samples (same budget)
  same wrong answer          P(two wrong samples give the same answer), within one strategy vs across strategies
  problem types              by all samples: every strategy solved it at least once / only some / none
  hv                         composition of the held-out set of phase-1 evaluations (stratum A: every base strategy
                             attempt failed, so strategy diversity is zero there by construction)

  python3 -m phase1.strat_overlap --phase0 runs/phase0 --phase1 runs/phase1 --hv runs/phase1_self/data/hv.jsonl
Writes <phase0>/strat_overlap.md and strat_overlap.json."""
import argparse
import gzip
import itertools
import json
import os
from math import comb

import numpy as np

K = 4
BOOT = 1000


def read_gz(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(l) for l in f]


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def add(per, r):
    """per[id][mode] -> list of (answer_key, correct); mode is 0..3 or 'neutral'."""
    m = r["mode"]
    if not (isinstance(m, int) and 0 <= m < K) and m != "neutral":
        return
    per.setdefault(r["id"], {}).setdefault(m, []).extend(zip(r["answers"], r["correct"]))


def load_pool(dirs):
    per = {}
    for d in dirs:
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if f.startswith("pool_") and f.endswith(".jsonl.gz"):
                for r in read_gz(os.path.join(d, f)):
                    add(per, r)
    return per


def load_test(phase0):
    td = os.path.join(phase0, "test")
    per = {}
    for name in [f"test_strategy{k}.jsonl.gz" for k in range(K)] + ["test_neutral.jsonl.gz"]:
        p = os.path.join(td, name)
        if os.path.exists(p):
            for r in read_gz(p):
                add(per, r)
    return per


# ----------------------------------------------------------------------------- per-problem statistics
def miss(n, c, m):
    """Unbiased P(m samples drawn without replacement from n, c of them correct, are all wrong)."""
    return comb(n - c, m) / comb(n, m) if n - c >= m else 0.0


def pairs_same(xs, ys=None):
    """(both wrong with an answer, same answer) over sample pairs: within xs, or across xs x ys."""
    it = itertools.combinations(xs, 2) if ys is None else itertools.product(xs, ys)
    both = same = 0
    for (a, ca), (b, cb) in it:
        if not ca and not cb and a and b:
            both += 1
            same += a == b
    return both, same


def problem_stats(d):
    """d[mode] -> [(answer, correct)]; None if some strategy has < 2 samples."""
    if any(len(d.get(k, [])) < 2 for k in range(K)):
        return None
    n = [len(d[k]) for k in range(K)]
    c = [sum(x[1] for x in d[k]) for k in range(K)]
    p = [c[k] / n[k] for k in range(K)]
    s = {"acc": p,
         "p2_same": float(np.mean([1 - miss(n[k], c[k], 2) for k in range(K)])),
         "p2_diff": float(np.mean([1 - (1 - p[a]) * (1 - p[b]) for a, b in itertools.combinations(range(K), 2)])),
         "p4_mix": 1 - float(np.prod([miss(n[k], c[k], 1) for k in range(K)])),
         "p8_mix": 1 - float(np.prod([miss(n[k], c[k], 2) for k in range(K)])),
         "solved_by": sum(ck > 0 for ck in c)}
    w = [pairs_same(d[k]) for k in range(K)]
    s["sw_same"] = (sum(x[0] for x in w), sum(x[1] for x in w))
    b = [pairs_same(d[a], d[b]) for a, b in itertools.combinations(range(K), 2)]
    s["sw_diff"] = (sum(x[0] for x in b), sum(x[1] for x in b))
    neu = d.get("neutral", [])
    if len(neu) >= 8:
        nn, cn = len(neu), sum(x[1] for x in neu)
        s.update(neu_acc=cn / nn, neu_p2=1 - miss(nn, cn, 2), neu_p4=1 - miss(nn, cn, 4), neu_p8=1 - miss(nn, cn, 8),
                 neu_solved=cn > 0, sw_neu=pairs_same(neu))
    return s


# ----------------------------------------------------------------------------- set summaries
def mean_of(S, key):
    return lambda idx: float(np.mean([S[i][key] for i in idx]))


def ratio_of(S, key):
    def f(idx):
        both = sum(S[i][key][0] for i in idx)
        return sum(S[i][key][1] for i in idx) / both if both else float("nan")
    return f


def diff(f, g):
    return lambda idx: f(idx) - g(idx)


def boot(fn, N, seed=0):
    rng = np.random.default_rng(seed)
    point = fn(np.arange(N))
    vals = [v for v in (fn(rng.integers(0, N, N)) for _ in range(BOOT)) if np.isfinite(v)]
    if not vals:
        return [point, float("nan"), float("nan")]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(point), float(lo), float(hi)]


def summarize(S):
    N = len(S)
    out = {"problems": N, "acc_per_strategy": [float(np.mean([s["acc"][k] for s in S])) for k in range(K)]}
    for key in ("p2_same", "p2_diff", "p4_mix", "p8_mix"):
        out[key] = mean_of(S, key)(np.arange(N))
    out["gain_p2"] = boot(diff(mean_of(S, "p2_diff"), mean_of(S, "p2_same")), N)
    out["sw_same"] = ratio_of(S, "sw_same")(np.arange(N))
    out["sw_diff"] = ratio_of(S, "sw_diff")(np.arange(N))
    out["sw_gap"] = boot(diff(ratio_of(S, "sw_same"), ratio_of(S, "sw_diff")), N)
    sb = np.array([s["solved_by"] for s in S])
    out["types"] = {"all": float((sb == K).mean()), "some": float(((sb > 0) & (sb < K)).mean()),
                    "none": float((sb == 0).mean())}
    if all("neu_p8" in s for s in S):
        for key in ("neu_acc", "neu_p2", "neu_p4", "neu_p8"):
            out[key] = mean_of(S, key)(np.arange(N))
        out["gain_p4"] = boot(diff(mean_of(S, "p4_mix"), mean_of(S, "neu_p4")), N)
        out["gain_p8"] = boot(diff(mean_of(S, "p8_mix"), mean_of(S, "neu_p8")), N)
        out["sw_neu"] = ratio_of(S, "sw_neu")(np.arange(N))
        neu = np.array([s["neu_solved"] for s in S])
        mix = np.array([s["p8_mix"] > 0 for s in S])      # any strategy sample correct (8 samples at n = 2)
        out["only_neutral"] = float((neu & ~mix).mean())
        out["only_strategies"] = float((~neu & mix).mean())
    return out


def by_source(per, ids, src):
    S = {}
    for i in ids:
        s = problem_stats(per[i])
        if s is not None:
            S.setdefault(src.get(i, "?"), []).append(s)
            S.setdefault("all", []).append(s)
    return {k: summarize(v) for k, v in S.items() if len(v) >= 20}


# ----------------------------------------------------------------------------- report
def f3(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:.3f}"


def ci(t):
    return "—" if not np.isfinite(t[0]) else f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def verdict(g):
    if not np.isfinite(g[1]):
        return "мало данных"
    if g[1] > 0.02:
        return "**другая стратегия заметно помогает**: способы дополняют друг друга"
    if g[0] < 0.01 and g[2] < 0.02:
        return "**другая стратегия почти не помогает**: способы решают и не решают одни и те же задачи"
    return "выигрыш небольшой (≈1–2 п.п.)"


def report(R, hv_meta):
    L = ["# Решают ли 4 стратегии одни и те же задачи? (базовая модель, данные фазы 0)\n",
         "Вопрос: если 4 способа решения почти всегда вместе решают и вместе не решают задачу, то модель с 4 режимами-"
         "«стратегиями» не может решить больше задач, чем одна модель с тем же числом попыток. Интервалы — 95% бутстрап "
         "по задачам.\n",
         "## 1. Главное: две попытки одной стратегией или двумя разными\n",
         "«Одна» — два решения одной стратегии; «разные» — по одному решению двух разных стратегий. Разница — сколько "
         "задач добавляет смена способа при том же числе попыток.\n",
         "| набор | задач | pass@2 одна | pass@2 разные | выигрыш | вывод |", "|---|---|---|---|---|---|"]
    for name, by in R.items():
        if "all" in by:
            s = by["all"]
            L.append(f"| {name} | {s['problems']} | {f3(s['p2_same'])} | {f3(s['p2_diff'])} | {ci(s['gain_p2'])} | "
                     f"{verdict(s['gain_p2'])} |")
    t = R.get("test", {}).get("all")
    if t and "gain_p8" in t:
        L += ["\n## 2. Тест фазы 0: 4 стратегии против одной температуры при равном числе попыток\n",
              "| | 4 попытки | 8 попыток |", "|---|---|---|",
              f"| температура (нейтральный промпт) | {f3(t['neu_p4'])} | {f3(t['neu_p8'])} |",
              f"| 4 стратегии (по 1 / по 2) | {f3(t['p4_mix'])} | {f3(t['p8_mix'])} |",
              f"| разница | {ci(t['gain_p4'])} | {ci(t['gain_p8'])} |",
              f"\nЗадач, где решила только температура (за 8 попыток): {f3(t['only_neutral'])}; только стратегии: "
              f"{f3(t['only_strategies'])}. pass@1 температуры {f3(t['neu_acc'])}, стратегий "
              f"{[round(x, 3) for x in t['acc_per_strategy']]}."]
    L += ["\n## 3. Одинаковые ошибки\n",
          "Доля пар неверных решений с **одинаковым** неверным ответом.\n",
          "| набор | внутри одной стратегии | между разными | разница | внутри температуры |", "|---|---|---|---|---|"]
    for name, by in R.items():
        if "all" in by:
            s = by["all"]
            L.append(f"| {name} | {f3(s['sw_same'])} | {f3(s['sw_diff'])} | {ci(s['sw_gap'])} | {f3(s.get('sw_neu'))} |")
    L += ["\n## 4. Типы задач (по всем решениям)\n",
          "| набор | все 4 стратегии решили хоть раз | только часть | ни одна |", "|---|---|---|---|"]
    for name, by in R.items():
        if "all" in by:
            ty = by["all"]["types"]
            L.append(f"| {name} | {f3(ty['all'])} | {f3(ty['some'])} | {f3(ty['none'])} |")
    L.append("\n«Только часть» включает случайность: при 2 решениях на стратегию задача с успехом 50% даёт 0 из 2 "
             "в четверти случаев. Честное сравнение — разделы 1–2.")
    L += ["\n## 5. По источникам: выигрыш pass@2 от смены стратегии\n", "| набор | источник | задач | выигрыш |",
          "|---|---|---|---|"]
    for name, by in R.items():
        for src, s in by.items():
            if src != "all":
                L.append(f"| {name} | {src} | {s['problems']} | {ci(s['gain_p2'])} |")
    if hv_meta:
        L += ["\n## 6. Набор hv (на нём оценивались режимы фазы 1)\n",
              f"Состав по слоям: {hv_meta}. Слой A — задачи, где **все** попытки базовой модели всеми 4 стратегиями "
              "неверны, поэтому разнообразие стратегий там по построению ничего не даёт. Слои B0/B12 — новые задачи, "
              "проверенные только нейтральным промптом: решений стратегиями на них нет."]
    L += ["\n## Как читать\n",
          "- Выигрыш в разделе 1 близок к нулю → способы решения у базовой модели не дополняют друг друга. Тогда "
          "неудивительно, что режимы, выучившие эти способы (`match`, `oracle`), не подняли pass@16: «разнообразие "
          "способов ≠ разнообразие ответов» уже у самих стратегий.",
          "- Выигрыш заметный, а у режимов фазы 1 его нет → потеря в обучении: режимы выучили стиль, но не то, что "
          "делает стратегию полезной на своих задачах. Тогда стоит чинить обучение.",
          "- Раздел 3: если неверные ответы между стратегиями совпадают почти так же часто, как внутри одной стратегии, "
          "ошибки общие (одна и та же ловушка в задаче), а не ошибки способа.",
          "- val выбран по успеху базы 10–90% (для него всё сдвинуто к середине); test не отобран — он главный."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase0", default="runs/phase0")
    ap.add_argument("--phase1", default="runs/phase1")
    ap.add_argument("--hv", default="runs/phase1_self/data/hv.jsonl")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or a.phase0
    src = {p["id"]: p.get("source", "?") for f in (f"{a.phase0}/data/pool.jsonl", f"{a.phase0}/data/test.jsonl",
                                                    f"{a.phase1}/hard/pool.jsonl") for p in load_jsonl(f)}
    R = {}
    test = load_test(a.phase0)
    if test:
        R["test"] = by_source(test, sorted(test), src)
    pool = load_pool([f"{a.phase0}/pool", f"{a.phase1}/hard/out"])
    if not pool and not test:
        raise SystemExit(f"no phase-0 outputs in {a.phase0}/test, {a.phase0}/pool or {a.phase1}/hard/out")
    val_ids = [p["id"] for p in load_jsonl(f"{a.phase1}/data/val.jsonl") if p["id"] in pool]
    if val_ids:
        R["val"] = by_source(pool, val_ids, src)
    if pool:
        R["pool"] = by_source(pool, sorted(pool), src)
    hv = load_jsonl(a.hv)
    hv_meta = {}
    for p in hv:
        hv_meta[p.get("stratum", "?")] = hv_meta.get(p.get("stratum", "?"), 0) + 1
    os.makedirs(out, exist_ok=True)
    md = os.path.join(out, "strat_overlap.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write(report(R, hv_meta))
    json.dump({"sets": R, "hv": hv_meta}, open(os.path.join(out, "strat_overlap.json"), "w"), indent=1, default=float)
    print(md)


if __name__ == "__main__":
    main()
