"""Phase-0 report: methods compared on the test set + pre-registered decision rules + pool stats.

Methods (M modes x S=2 samples per problem):
  temperature : neutral prompt; modes = samples 0..3, second sample of mode i = sample 4+i;
                one neutral verifier (same for every j)
  strategies  : 4 system-prompt strategies; verifier j = strategy-specific checker j
  noise       : M noisy copies of the base weights; verifier j = noisy copy j
"""
import gzip
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pilot.metrics import auroc, per_problem, summarize  # noqa: E402

from .data import load_jsonl  # noqa: E402

# decision thresholds -- fixed before looking at the results
TH = {"phi_gap": 0.05, "same_wrong_gap": 0.05, "auroc_min": 0.60, "exactly_one_min": 0.05,
      "acc_lo": 0.30, "acc_hi": 0.85}
BOOT = 500


def read_gz(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(l) for l in f]


def arrays(test, sol, p_no):
    """sol[i][s] -> list over problems of (answer_key, correct, text); p_no[i][j] -> list over problems."""
    M, S, N = len(sol), len(sol[0]), len(test)
    A = {"correct": np.zeros((M, S, N), bool), "answer": np.full((M, S, N), -1, np.int64)}
    truth = np.zeros(N, np.int64)
    bins = sorted({p["bin"] for p in test})
    A["op"] = np.array([bins.index(p["bin"]) for p in test])
    A["sol_hash"] = np.zeros((M, N), np.int64)
    for n in range(N):
        ids = {"__truth__": 0}
        for i in range(M):
            for s in range(S):
                k, c, _ = sol[i][s][n]
                A["correct"][i, s, n] = c
                if k:
                    A["answer"][i, s, n] = 0 if c else ids.setdefault(k, len(ids))
        truth[n] = 0
        for i in range(M):
            A["sol_hash"][i, n] = hash(sol[i][0][n][2])
    A["truth"] = truth
    A["has_err"] = ~A["correct"]
    A["first_err"] = np.zeros((M, S, N), np.int64)
    A["err_pid"] = np.full((M, S, N), -1, np.int64)
    A["backward"] = np.zeros((M, S, N), bool)
    A["p_err"] = np.array(p_no, dtype=float)                  # [I,J,N]
    A["loc"] = np.zeros((M, M, N), np.int64)
    ans0, ans1 = A["answer"][:, 0], A["answer"][:, 1]
    A["cmp_flag"] = np.zeros((M, M, N), bool)
    for i in range(M):
        for j in range(M):
            other = ans0[j] if i != j else ans1[i]
            A["cmp_flag"][i, j] = (ans0[i] != other) | (ans0[i] < 0)
    A["cmp_loc"] = np.zeros((M, M, N), np.int64)
    return A, bins


def load_methods(run):
    test = load_jsonl(os.path.join(run, "data", "test.jsonl"))
    methods = {}
    td = os.path.join(run, "test")
    f_neu, f_vn = os.path.join(td, "test_neutral.jsonl.gz"), os.path.join(td, "verify_neutral.json")
    if os.path.exists(f_neu) and os.path.exists(f_vn):
        neu = read_gz(f_neu)
        sol = [[[(r["answers"][i + 4 * s], r["correct"][i + 4 * s], r["texts"][i + 4 * s]) for r in neu]
                for s in range(2)] for i in range(4)]
        v = json.load(open(f_vn))
        p = [[v["p_no"][i]] * 4 for i in range(4)]
        methods["temperature"] = arrays(test, sol, p) + (v["missing"],)
    f_vs = os.path.join(td, "verify_strategies.json")
    if os.path.exists(f_vs):
        st = [read_gz(os.path.join(td, f"test_strategy{k}.jsonl.gz")) for k in range(4)]
        sol = [[[(r["answers"][s], r["correct"][s], r["texts"][s]) for r in st[i]] for s in range(2)]
               for i in range(4)]
        v = json.load(open(f_vs))
        methods["strategies"] = arrays(test, sol, v["p_no"]) + (v["missing"],)
    nd = os.path.join(run, "noise")
    k = 0
    while os.path.exists(os.path.join(nd, f"solve_{k}.jsonl.gz")):
        k += 1
    if k >= 2 and all(os.path.exists(os.path.join(nd, f"verify_{j}.json")) for j in range(k)):
        so = [read_gz(os.path.join(nd, f"solve_{i}.jsonl.gz")) for i in range(k)]
        sol = [[[(r["answers"][s], r["correct"][s], r["texts"][s]) for r in so[i]] for s in range(2)]
               for i in range(k)]
        vs = [json.load(open(os.path.join(nd, f"verify_{j}.json"))) for j in range(k)]
        p = [[vs[j]["p_no"][i] for j in range(k)] for i in range(k)]
        methods["noise"] = arrays(test, sol, p) + (sum(v["missing"] for v in vs),)
    return test, methods


# ----------------------------------------------------------------------------- per-problem stats
def phi_within(wrong, op, idx):
    """Mean pairwise phi of error indicators within difficulty bins, on problems idx."""
    M = wrong.shape[0]
    w, o = wrong[:, idx], op[idx]
    vals = []
    for b in np.unique(o):
        m = o == b
        if m.sum() < 15:
            continue
        for i in range(M):
            for j in range(i + 1, M):
                x, y = w[i, m].astype(float), w[j, m].astype(float)
                if x.std() > 0 and y.std() > 0:
                    vals.append(np.corrcoef(x, y)[0, 1])
    return float(np.mean(vals)) if vals else float("nan")


def same_wrong(A, idx):
    ans, cor = A["answer"][:, 0][:, idx], A["correct"][:, 0][:, idx]
    M = ans.shape[0]
    both = same = 0
    for i in range(M):
        for j in range(i + 1, M):
            m = ~cor[i] & ~cor[j] & (ans[i] > 0) & (ans[j] > 0)
            both += m.sum()
            same += (ans[i][m] == ans[j][m]).sum()
    return float(same / both) if both else float("nan")


def boot_ci(fn, N, seed=0):
    rng = np.random.default_rng(seed)
    point = fn(np.arange(N))
    vals = [fn(rng.integers(0, N, N)) for _ in range(BOOT)]
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def fmt(x, d=3):
    return "—" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def ci(t):
    return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]" if np.isfinite(t[0]) else "—"


# ----------------------------------------------------------------------------- pool
def numbers(text):
    return set(re.findall(r"\d+(?:\.\d+)?", text or ""))


def n_clusters(texts, thr=0.8):
    reps = []
    for t in texts:
        s = numbers(t)
        if not any(len(s | r) and len(s & r) / len(s | r) >= thr for r in reps):
            reps.append(s)
    return len(reps)


def pool_stats(run):
    files = sorted(f for f in os.listdir(os.path.join(run, "pool"))) if os.path.isdir(os.path.join(run, "pool")) else []
    rows = [r for f in files if f.endswith(".jsonl.gz") for r in read_gz(os.path.join(run, "pool", f))]
    if not rows:
        return None
    pool = {p["id"]: p for p in load_jsonl(os.path.join(run, "data", "pool.jsonl"))}
    per = {}
    for r in rows:
        d = per.setdefault(r["id"], {"correct": [], "texts_ok": [], "wrong": 0})
        d["correct"] += r["correct"]
        d["texts_ok"] += [t for t, c in zip(r["texts"], r["correct"]) if c]
        d["wrong"] += sum((not c) and a != "" for c, a in zip(r["correct"], r["answers"]))
    by_src = {}
    for pid, d in per.items():
        src = pool[pid]["source"] if pid in pool else "?"
        by_src.setdefault(src, []).append(d)
    out = {}
    for src, ds in list(by_src.items()) + [("all", [d for v in by_src.values() for d in v])]:
        k = np.array([sum(d["correct"]) for d in ds])
        n = np.array([len(d["correct"]) for d in ds])
        cl = [n_clusters(d["texts_ok"]) for d in ds if d["texts_ok"]]
        out[src] = {"problems": len(ds), "acc": float(k.sum() / max(n.sum(), 1)),
                    "share_any_correct": float((k > 0).mean()),
                    "share_stage2_useful": float(((k >= 1) & (k <= n - 2)).mean()),
                    "hist": np.bincount(k, minlength=int(n.max()) + 1).tolist(),
                    "distinct_correct_mean": float(np.mean(cl)) if cl else 0.0,
                    "natural_errors": int(sum(d["wrong"] for d in ds))}
    return out


# ----------------------------------------------------------------------------- report
def build(run):
    state = json.load(open(os.path.join(run, "state.json"))) if os.path.exists(os.path.join(run, "state.json")) else {}
    test, methods = load_methods(run)
    N = len(test)
    S, V = {}, {}
    for name, (A, bins, miss) in methods.items():
        S[name] = summarize(A)
        S[name]["missing_yes_no"] = miss
        V[name] = per_problem(A)
    L = ["# Фаза 0: Qwen3 без обучения — отчёт\n"]
    L.append(f"- модель: `{state.get('model')}`, бэкенд: {state.get('backend')}, задач в тесте: {N}, "
             f"время: {state.get('hours', 0):.2f} ч" + (" — **SMOKE-прогон, числа не значимы**" if state.get("smoke") else ""))
    for n in state.get("data_notes", []):
        L.append(f"- данные: {n}")
    if state.get("noise"):
        z = state["noise"]
        L.append(f"- шум: точность на val без шума {z['base_val_acc']:.3f}, по σ {z['val_acc']} → σ = {z['sigma']} {z.get('note', '')}")
    if state.get("failed"):
        L.append(f"- **не выполнены шаги:** {state['failed']} (см. logs/)")

    # ---------------- decisions
    D = []
    have = lambda *ms: all(m in methods for m in ms)
    if have("temperature", "strategies"):
        At, As = methods["temperature"][0], methods["strategies"][0]
        wt, ws, op = ~At["correct"][:, 0], ~As["correct"][:, 0], At["op"]
        d_phi = boot_ci(lambda i: phi_within(wt, op, i) - phi_within(ws, op, i), N)
        d_sw = boot_ci(lambda i: same_wrong(At, i) - same_wrong(As, i), N)
        if not (np.isfinite(d_phi[0]) or np.isfinite(d_sw[0])):
            verdict = "❔ мало ошибок для оценки — увеличьте тест или возьмите задачи потруднее"
        elif (np.isfinite(d_phi[0]) and d_phi[0] >= TH["phi_gap"]) or \
                (np.isfinite(d_sw[0]) and d_sw[0] >= TH["same_wrong_gap"]):
            verdict = "⚠️ промпты заметно разводят ошибки → **планка для фазы 1: обогнать 4 промпта, а не температуру**"
        else:
            verdict = "✅ промпты почти не разводят ошибки → **есть что улучшать, идём в фазу 1**"
        D.append(("1. Есть что улучшать?",
                  f"снижение φ ошибок промптами относительно температуры: {ci(d_phi)}; снижение доли одинаковых "
                  f"неверных ответов: {ci(d_sw)}", verdict))
        Ms = As["correct"].shape[0]
        exactly = lambda A, i: float((A["correct"][:, 0][:, i].sum(0) == 1).mean())
        e1 = boot_ci(lambda i: exactly(As, i), N)
        D.append(("3. Сигнал для этапа 2",
                  f"доля задач, где верна ровно одна стратегия из {Ms}: {ci(e1)}",
                  "✅ достаточно" if e1[0] >= TH["exactly_one_min"] else
                  f"⚠️ меньше {TH['exactly_one_min']:.0%} → для этапа 2 нужны задачи потруднее"))
        gain = boot_ci(lambda i: V["strategies"]["vvote_cross"][i].mean() - V["temperature"]["maj4"][i].mean(), N)
        gmaj = boot_ci(lambda i: V["strategies"]["maj4"][i].mean() - V["temperature"]["maj4"][i].mean(), N)
        D.append(("5. Насколько силён конкурент «4 промпта»",
                  f"maj@4 стратегий − maj@4 температуры: {ci(gmaj)}; голосование с перекрёстной проверкой "
                  f"стратегий − maj@4 температуры: {ci(gain)}",
                  "⚠️ промпты уже улучшают итоговую точность — сильный конкурент" if gain[1] > 0 or gmaj[1] > 0 else
                  "✅ промпты не дают значимого выигрыша в итоговой точности"))
    if have("temperature"):
        At = methods["temperature"][0]
        lab = At["has_err"][:, 0]
        sc = At["p_err"][:, 0]
        au = boot_ci(lambda i: auroc(sc[:, i].ravel(), lab[:, i].ravel()), N)
        D.insert(1, ("2. Работает ли проверка по промпту",
                     f"AUROC нейтрального проверяющего на решениях с температурой: {ci(au)}",
                     "✅ работает" if au[0] >= TH["auroc_min"] else
                     f"⚠️ AUROC < {TH['auroc_min']} → **в фазе 1 обучить проверяющего на естественных ошибках** "
                     "(данные из пула)"))
        acc = S["temperature"]["pass1"]
        bins = methods["temperature"][1]
        per_bin = {b: float(At["correct"][:, 0][:, At["op"] == k].mean()) for k, b in enumerate(bins)}
        src = {}
        for k, b in enumerate(bins):
            src.setdefault(b.split("_")[0], []).append(At["correct"][:, 0][:, At["op"] == k].ravel())
        src = {s: float(np.concatenate(v).mean()) for s, v in src.items()}
        bad = [s for s, v in src.items() if not TH["acc_lo"] <= v <= TH["acc_hi"]]
        D.append(("4. Подходит ли сложность",
                  f"pass@1 (температура) всего {acc:.3f}; по источникам {({k: round(v, 3) for k, v in src.items()})}",
                  "✅ все источники в диапазоне" if not bad else
                  f"⚠️ вне диапазона [{TH['acc_lo']}, {TH['acc_hi']}]: {bad} → поменять задачи этих источников"))
    D.sort(key=lambda d: d[0])
    L.append("\n## 1. Решения по заранее зафиксированным порогам\n")
    L.append(f"Пороги: {TH}. Интервалы — 95% бутстрап по задачам.\n")
    L.append("| вопрос | измерение | вывод |")
    L.append("|---|---|---|")
    for q, m, c in D:
        L.append(f"| {q} | {m} | {c} |")

    L.append("\n## 2. Точность и совпадение ошибок\n")
    L.append("| метод | pass@1 | maj@M | pass@M | pass@2M | φ ошибок внутри сложности | одинаковый неверный ответ | "
             "ровно один режим прав | разные ответы у режимов |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for name, (A, bins, _) in methods.items():
        s = S[name]
        M = A["correct"].shape[0]
        ex1 = float((A["correct"][:, 0].sum(0) == 1).mean())
        L.append(f"| {name} | {fmt(s['pass1'])} | {fmt(s['maj4'])} | {fmt(s.get(f'pass@{M}'))} | "
                 f"{fmt(s.get(f'pass@{2 * M}'))} | {fmt(phi_within(~A['correct'][:, 0], A['op'], np.arange(N)))} | "
                 f"{fmt(same_wrong(A, np.arange(N)))} | {fmt(ex1)} | {fmt(s['modes_differ_answer'])} |")
    L.append("\nТочность по режимам (решение 0):\n")
    for name, (A, bins, _) in methods.items():
        L.append(f"- {name}: {[round(x, 3) for x in S[name]['acc_per_mode']]}")
    if "temperature" in methods:
        A, bins, _ = methods["temperature"]
        L.append("\nТочность по группам сложности (температура): " +
                 ", ".join(f"{b} {A['correct'][:, 0][:, A['op'] == k].mean():.3f}" for k, b in enumerate(bins)))

    L.append("\n## 3. Кто кого проверяет\n")
    L.append("Протокол 1: проверяющий читает решение и отвечает Yes/No (оценка P(No)); порог на каждую ячейку "
             "(решающий i, проверяющий j) подобран так, чтобы FPR ≈ 5%. «Диаг.» — тот же режим, «вне» — другой. "
             "Протокол 2: «реши сам и сравни ответ»; на диагонали — второй сэмпл того же режима.\n")
    L.append("| метод | ошибок | AUROC диаг. / вне | TPR@5%FPR диаг. / вне | сравнение: TPR диаг. / вне | "
             "сравнение: FPR диаг. / вне | нет Yes/No в top-20 |")
    L.append("|---|---|---|---|---|---|---|")
    for name in methods:
        s = S[name]
        L.append(f"| {name} | {s['n_errors_total']} | {fmt(s['auroc_diag'])} / {fmt(s['auroc_off'])} | "
                 f"{fmt(s['ver_tpr_diag'])} / {fmt(s['ver_tpr_off'])} | {fmt(s['cmp_tpr_diag'])} / {fmt(s['cmp_tpr_off'])} | "
                 f"{fmt(s['cmp_fpr_diag'])} / {fmt(s['cmp_fpr_off'])} | {s['missing_yes_no']} |")
    for name in ("strategies", "noise"):
        if name in S:
            L.append(f"\nAUROC-матрица {name} (строка — решающий, столбец — проверяющий): "
                     f"{np.round(np.array(S[name]['auroc_matrix']), 3).tolist()}")

    L.append("\n## 4. Итоговая точность процедур\n")
    L.append("| метод | одно решение | maj@M | голосование + самопроверка | голосование + перекрёстная проверка | "
             "замена при тревоге (+исправлено / −испорчено) |")
    L.append("|---|---|---|---|---|---|")
    for name in methods:
        s = S[name]
        L.append(f"| {name} | {fmt(s['pass1'])} | {fmt(s['maj4'])} | {fmt(s['vvote_self'])} | {fmt(s['vvote_cross'])} | "
                 f"{fmt(s['switch_acc'])} (+{fmt(s['switch_fixed'])} / −{fmt(s['switch_broken'])}) |")
    L.append("\nУ температуры проверяющий один, поэтому «самопроверка» = «перекрёстная». У стратегий и шума веса "
             "голосования — сырые P(No) разных проверяющих, они могут быть по-разному откалиброваны.")

    ps = pool_stats(run)
    if ps:
        L.append("\n## 5. Пул для фазы 1 (4 стратегии × 2 решения на задачу)\n")
        L.append("| источник | задач | точность | есть ≥1 верное | годится для этапа 2 (верно от 1 до n−2) | "
                 "структурно разных верных решений (в среднем) | естественных ошибок | гистограмма числа верных |")
        L.append("|---|---|---|---|---|---|---|---|")
        for src, d in ps.items():
            L.append(f"| {src} | {d['problems']} | {d['acc']:.3f} | {d['share_any_correct']:.3f} | "
                     f"{d['share_stage2_useful']:.3f} | {d['distinct_correct_mean']:.2f} | {d['natural_errors']} | {d['hist']} |")
        L.append("\n«Структурно разные» — кластеры верных решений по набору встречающихся чисел (Жаккар < 0.8).")

    L.append("\n## Как читать\n")
    L.append("- Раздел 1 — главное: каждая строка говорит, что делать в фазе 1.\n"
             "- Если в п. 1 ✅, а в п. 5 ⚠️ — промпты не разводят ошибки по φ, но выигрывают в точности: сравнивать "
             "фазу 1 нужно с ними.\n"
             "- Если в п. 2 ⚠️ — матрица «кто кого проверяет» в разделе 3 по протоколу 1 мало что значит; смотрите протокол 2.\n"
             "- Пул (раздел 5) — данные для multi-reference WTA: нужны задачи с несколькими структурно разными "
             "верными решениями и с долей «годится для этапа 2» хотя бы 20–30%.")
    out = os.path.join(run, "report_phase0.md")
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    json.dump({"summary": {k: {kk: vv for kk, vv in v.items()} for k, v in S.items()}, "pool": ps, "state": state},
              open(os.path.join(run, "summary_phase0.json"), "w"), default=float, indent=1)
    return out


if __name__ == "__main__":
    print(build(sys.argv[1]))
