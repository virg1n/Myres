"""Report of the continuous-z run (phase1/zrun.py): pass@k up to k = 32, greedy answers with different z, paired
bootstrap over problems (seed 0 only: the interval covers problems, not trainings).

pass@k is estimated from all attempts without replacement (Chen et al. 2021): one stratum of k samples for continuous
z and for the model without modes; for 4 modes the k attempts are spread evenly over the modes (k / 4 each; for
k < 4 a random set of k modes), as the modes would be used. maj@k: majority vote over k such attempts, averaged
over random draws; a tie of m leaders counts 1/m if the right answer is among them (no order of modes favoured)."""
import argparse
import gzip
import itertools
import json
import os
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase0.answers import key as akey  # noqa: E402
from phase0.report import check_ids, fmt  # noqa: E402
from phase1 import film as FM  # noqa: E402
from phase1.zrun import ZDIMS, refs  # noqa: E402

NEW = [(f"wtaz_d{d}", f"непрерывный z, d = {d}") for d in ZDIMS]
REF = [("wta1", "4 заученных кода, WTA"), ("loramcl", "LoRA-MCL, 4 отдельные LoRA"),
       ("sftmulti", "SFT-multi, без режимов")]
KS = (1, 2, 4, 8, 16, 32)
PAIR_METRICS = [("pass1", "pass@1"), ("pass4", "pass@4"), ("pass32", "pass@32"), ("maj32", "maj@32"),
                ("gpass4", "pass@4 жадно")]


def read_gz(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(l) for l in f]


def load(d, test):
    """dict(kind, C [S, n, N] bool, A [S, n, N] int (-1 = no answer), truth [N], greedy [(correct [N], answer [N])])
    or None if the evaluation is not finished. ValueError if a file does not follow the test set."""
    mf = os.path.join(d, "eval.json")
    if not os.path.exists(mf):
        return None
    man = json.load(open(mf))
    N = len(test)
    ids = [{} for _ in range(N)]
    truth = np.array([ids[j].setdefault(akey(p["answer"]), 0) for j, p in enumerate(test)])

    def enc(rows, path):
        check_ids(rows, test, path)
        n = len(rows[0]["answers"])
        if any(len(r["answers"]) != n or len(r["correct"]) != n for r in rows):
            raise ValueError(f"{os.path.basename(path)}: a different number of samples per problem")
        cor = np.array([r["correct"] for r in rows], dtype=bool).T                       # [n, N]
        ans = np.array([[ids[j].setdefault(x, len(ids[j])) if x else -1 for x in r["answers"]]
                        for j, r in enumerate(rows)]).T
        return cor, ans

    units = [enc(read_gz(os.path.join(d, f"solve_{t}_test.jsonl.gz")), f"solve_{t}") for t in range(man["units"])]
    C, A = np.stack([u[0] for u in units]), np.stack([u[1] for u in units])
    if man["kind"] == "z":                       # z attempts are exchangeable: one stratum
        C, A = C.reshape(1, -1, N), A.reshape(1, -1, N)
    if C.shape[0] * C.shape[1] != man["k"]:
        raise ValueError(f"{d}: {C.shape[0] * C.shape[1]} attempts, eval.json says k = {man['k']}")
    greedy = [tuple(x[0] for x in enc(read_gz(os.path.join(d, f"greedy_{t}_test.jsonl.gz")), f"greedy_{t}"))
              for t in man["greedy"]]
    return {"kind": man["kind"], "C": C, "A": A, "truth": truth, "greedy": greedy}


def pass_at(C, k):
    """C [S, n, N] -> P(some of k attempts is correct) per problem: k // S attempts from every stratum and one more
    from k % S random strata, without replacement inside a stratum."""
    S, n, N = C.shape
    base, r = divmod(k, S)
    if base + (r > 0) > n:
        raise ValueError(f"k = {k} needs more than {n} samples per stratum")
    c = C.sum(1)

    def miss(a):                       # P(a draws of a stratum are all wrong) = C(n - c, a) / C(n, a)
        q = np.ones((S, N))
        for i in range(a):
            q *= np.clip(n - c - i, 0, None) / (n - i)
        return q

    q0, q1 = miss(base), miss(base + 1) if r else None
    subsets = list(itertools.combinations(range(S), r))
    allwrong = sum(np.prod([q1[s] if s in R else q0[s] for s in range(S)], axis=0) for R in subsets)
    return 1 - allwrong / len(subsets)


def vote(rows, truth):
    """rows [k, N] int (-1 = no answer) -> per problem 1 / (number of tied leaders) if the truth is a leader, else 0."""
    out = np.zeros(rows.shape[1])
    for j in range(rows.shape[1]):
        a = rows[:, j][rows[:, j] >= 0]
        if len(a):
            vals, cnt = np.unique(a, return_counts=True)
            lead = vals[cnt == cnt.max()]
            out[j] = (truth[j] in lead) / len(lead)
    return out


def maj_at(A, truth, k, rng, draws=200):
    """P(majority answer of k attempts is right) per problem, attempts drawn as in pass_at."""
    S, n, N = A.shape
    base, r = divmod(k, S)
    if base == n and r == 0:           # all attempts: nothing to draw
        return vote(A.reshape(-1, N), truth)
    out = np.zeros(N)
    for _ in range(draws):
        extra = set(rng.choice(S, r, replace=False).tolist())
        rows = [A[s, i] for s in range(S) for i in rng.choice(n, base + (s in extra), replace=False)]
        out += vote(np.stack(rows), truth)
    return out / draws


def vectors(E, rng):
    """Per-problem vectors of every metric (None where it does not apply)."""
    C, A, G = E["C"], E["A"], E["greedy"]
    S, n, N = C.shape
    v = {f"pass{k}": pass_at(C, k) for k in KS if k <= S * n}
    v["maj4"], v["maj32"] = maj_at(A, E["truth"], 4, rng), maj_at(A, E["truth"], S * n, rng)
    flat = A.reshape(-1, N)
    v["unique"] = np.array([len(set(flat[:, j][flat[:, j] >= 0].tolist())) for j in range(N)], dtype=float)
    for g in (1, 4, 8):
        if len(G) >= g:
            cor, ans = np.stack([x[0] for x in G[:g]]), np.stack([x[1] for x in G[:g]])
            v[f"gpass{g}"] = cor.any(0).astype(float)
            if g > 1:
                v[f"gdiffer{g}"] = (ans != ans[0]).any(0).astype(float)
                v[f"gunique{g}"] = np.array([len(set(ans[:, j].tolist())) for j in range(N)], dtype=float)
    if G:
        v["gmean1"] = np.stack([x[0] for x in G]).mean(0)
    return v


def gamma_spread(adapter, base):
    """(mean over coordinates of the SD of gamma between z (256 draws) or codes, mean |gamma|), or None."""
    if not os.path.exists(adapter):
        return None
    ck = torch.load(adapter, map_location="cpu", weights_only=False)
    from transformers import AutoConfig
    mc = AutoConfig.from_pretrained(base)
    film = FM.load_film(ck["cfg"], ck["adapter"], 2 * mc.num_hidden_layers + 1, mc.hidden_size)
    if film is None:
        return None
    with torch.no_grad():
        g = film.gamma_z(film.sample_z(256, torch.Generator().manual_seed(0))) if film.codes is None \
            else film.gamma(torch.arange(film.K))
    return float(g.std(0).mean()), float(g.abs().mean())


def build(run, data, base, phase1, seeds_run, boot=1000):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    test = [json.loads(l) for l in open(os.path.join(data, "test.jsonl"))]
    N = len(test)
    rd = refs(SimpleNamespace(phase1=phase1, seeds_run=seeds_run))
    tdir = {**{v: os.path.join(run, "train", v) for v, _ in NEW}, **rd}
    rng = np.random.default_rng(0)
    vec, kinds, missing = {}, {}, []
    for v, _ in NEW + REF:
        try:
            E = load(os.path.join(run, "eval", v), test)
        except (ValueError, OSError) as e:
            missing.append(f"{v} (отклонён: {e})")
            continue
        if E is None:
            missing.append(v)
            continue
        vec[v], kinds[v] = vectors(E, rng), E["kind"]
    st = json.load(open(os.path.join(run, "state.json"))) if os.path.exists(os.path.join(run, "state.json")) else {}
    k = st.get("k", 32)

    L = ["# Фаза 1: непрерывный случайный z вместо 4 заученных кодов\n"]
    L.append(f"- задач в тесте: {N}; seed 0, по одному обучению на вариант; k = {k} попыток на задачу.")
    L.append("- Непрерывный z: попытка j каждой задачи решается с z_j; z_0, z_1, … свои для каждой попытки, общие для "
             "задач (одна экспортированная модель на z). Режимы: по k / 4 попыток на режим. SFT-multi: k сэмплов.")
    if missing:
        L.append(f"- **нет данных:** {', '.join(missing)}")
    bad = {j: s["status"] for j, s in st.get("jobs", {}).items() if s.get("status") in ("failed", "skipped")}
    if bad:
        L.append(f"- **не выполнены:** {bad} (см. logs/)")

    def row(name, desc, keys):
        return f"| {name} ({desc}) | " + " | ".join(fmt(float(vec[name][x].mean())) if x in vec[name] else "—"
                                                     for x in keys) + " |"

    L.append("\n## 1. pass@k и голосование\n")
    keys = [f"pass{k_}" for k_ in KS if k_ <= k] + ["maj4", "maj32", "unique"]
    L.append("| метод | " + " | ".join(f"pass@{k_}" for k_ in KS if k_ <= k) + f" | maj@4 | maj@{k} | "
             f"разных ответов из {k} |")
    L.append("|---|" + "---|" * len(keys))
    for name, desc in NEW + REF:
        if name in vec:
            L.append(row(name, desc, keys))

    L.append("\n## 2. Жадное решение (T = 0) с разными z / режимами\n")
    L.append("Если z ничего не меняет, жадные ответы с разными z совпадают почти везде, и разнообразие даёт только "
             "сэмплирование токенов.\n")
    gk = ["gmean1", "gpass4", "gdiffer4", "gunique4", "gpass8", "gdiffer8", "gunique8"]
    L.append("| метод | pass@1 жадно | pass@4 жадно | ответы различаются (из 4) | разных ответов (из 4) | "
             "pass@8 жадно | различаются (из 8) | разных (из 8) |")
    L.append("|---|" + "---|" * len(gk))
    for name, desc in NEW + REF:
        if name in vec:
            L.append(row(name, desc, gk))

    L.append("\n## 3. Парные сравнения, X − R\n")
    L.append("Ячейка: разность [95% бутстрап по задачам]. Интервал условный: при этих обученных моделях и этом "
             "наборе z и сэмплов. Разброс между обучениями (один seed) и между наборами z он не учитывает.\n")
    L.append("| X − R | " + " | ".join(n for _, n in PAIR_METRICS) + " |")
    L.append("|---|" + "---|" * len(PAIR_METRICS))
    idxs = [rng.integers(0, N, N) for _ in range(boot)]
    comps = {}
    pairs = [(x, r) for x, _ in NEW for r, _ in REF] + [(f"wtaz_d{d}", "wtaz_d16") for d in ZDIMS if d != 16]
    for X, R in pairs:
        if X not in vec or R not in vec:
            continue
        cells = []
        for m, _ in PAIR_METRICS:
            if m not in vec[X] or m not in vec[R]:
                cells.append("—")
                continue
            dv = vec[X][m] - vec[R][m]
            reps = np.array([dv[i].mean() for i in idxs])
            lo, hi = np.percentile(reps, [2.5, 97.5])
            comps[f"{X}-{R}:{m}"] = {"mean": float(dv.mean()), "lo": float(lo), "hi": float(hi)}
            cells.append(f"{dv.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
        L.append(f"| {X} − {R} | " + " | ".join(cells) + " |")

    L.append("\n## 4. Обучение\n")
    L.append("| вариант | z | шагов | эпох | токенов | часов | выигрыш лучшего z: всё / последние 20% | "
             "разброс γ между z (кодами) | средний abs(γ) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for name, _ in NEW + REF:
        f = os.path.join(tdir[name], "result.json")
        if not os.path.exists(f):
            continue
        r = json.load(open(f))
        gs = gamma_spread(os.path.join(tdir[name], "adapter.pt"), base)
        zg = f"{r['z_gain']:.4f} / {fmt(r.get('z_gain_late'), 4)}" if "z_gain" in r else "—"
        L.append(f"| {name} | {r.get('zdim', 'коды' if gs else '—')} | {r.get('steps_done')} | "
                 f"{r.get('epochs_done', 0):.2f} | {r['tokens'] / 1e6:.1f}M | {r['hours']:.2f} | {zg} | "
                 f"{fmt(gs[0], 4) if gs else '—'} | {fmt(gs[1], 4) if gs else '—'} |")

    L.append("\n## Как читать\n")
    L.append("- Главный вопрос — растёт ли pass@k непрерывного z при больших k быстрее, чем у 4 кодов (wta1) и у "
             "температуры (SFT-multi): сравните строки раздела 1 при k = 4 и k = 32 и столбец pass@32 раздела 3.\n"
             "- Раздел 2 проверяет, что z вообще используется: при T = 0 разные z должны давать разные ответы хотя "
             "бы на части задач. «Выигрыш лучшего z» в разделе 4 — насколько NLL лучшего из 4 кандидатов ниже "
             "среднего (на токен); около 0 — модель игнорирует z. Оба показывают, что z влияет, но не что это "
             "полезное разнообразие: неверные ответы тоже могут различаться.\n"
             "- Веса норм в модели для vLLM хранятся в bf16 (шаг ~0.4% веса): если разброс γ между z в разделе 4 "
             "порядка 0.004 и меньше, часть различий между z при выгрузке округляется (у кодов wta1 так же).\n"
             "- z = N(0, I) / √d подаётся в FiLM умноженным на 4 = √16, первый слой — как у wta1 (граница "
             "инициализации 1/4). При d = 16 это параметризация wta1 со свежими кодами вместо заученных; при любом d "
             "W·z стартует с той же дисперсией, и шаг Adam сдвигает его одинаково. Скрытый слой FiLM — 64 нейрона, "
             "поэтому при d = 128 и 1024 γ лежит в семействе размерности ≤ 64: эти варианты отличаются "
             "параметризацией, а не числом степеней свободы.\n"
             "- wtaz отличается от wta1 не только непрерывностью z: коды не учатся и нет балансировки ёмкости (у "
             "кандидатов нет общих меток), исследование eps то же. Это сравнение двух схем целиком при равном времени "
             "обучения, а не чистый эффект «бесконечного числа режимов».\n"
             "- Одно обучение на вариант: выбор лучшего d по этому тесту — разведка, его стоит подтвердить другим seed.")
    out = os.path.join(run, "report_z.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    summ = {v: {m: float(x.mean()) for m, x in d.items()} for v, d in vec.items()}
    json.dump({"summary": summ, "kinds": kinds, "comparisons": comps, "missing": missing},
              open(os.path.join(run, "summary_z.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--phase1", required=True)
    ap.add_argument("--seeds_run", required=True)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--mark", default="")
    a = ap.parse_args()
    print(build(a.run, a.data, a.base, a.phase1, a.seeds_run, a.boot))
    if a.mark:
        os.makedirs(a.mark, exist_ok=True)
        open(os.path.join(a.mark, "DONE"), "w").close()
