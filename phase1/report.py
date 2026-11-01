"""Phase-1 report: trained modes vs phase-0 baselines on the same test problems (paired)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase0.report import boot_ci, ci, fmt, load_methods, phi_within, same_wrong  # noqa: E402
from phase1.metrics import graded_at, graded_auroc, load_eval  # noqa: E402
from pilot.metrics import per_problem, summarize  # noqa: E402

VARIANTS = [("sft1", "SFT, 1 эталон"), ("sftmulti", "SFT, несколько эталонов"),
            ("filmmean", "FiLM без WTA"), ("wta1", "WTA, 1 эталон"), ("wtamulti", "WTA, несколько эталонов"),
            ("wtabal", "WTA, несколько эталонов + паросочетание"),
            ("s2_best", "этап 2 на лучшем варианте"), ("s2_ctrl", "этап 2 на SFT-multi (контроль)")]
TH = {"pass_margin": 0.02}


def load_all(run, phase0, noise2):
    test = [json.loads(l) for l in open(os.path.join(run, "data", "test.jsonl"))]
    methods, desc = {}, {}
    if phase0 and os.path.isdir(phase0):
        _, m0 = load_methods(phase0)
        for k, v in m0.items():
            if k == "noise":       # phase-0 noise members shared one sampling seed; noise2 replaces it
                continue
            methods[f"p0_{k}"] = v + (True,)
            desc[f"p0_{k}"] = {"temperature": "фаза 0: температура", "strategies": "фаза 0: 4 стратегии",
                               "noise": "фаза 0: шум σ=0.02"}[k]
    if noise2 and os.path.exists(os.path.join(noise2, "noise", "verify_0.json")):
        _, mn = load_methods(noise2)
        if "noise" in mn:
            methods["noise_big"] = mn["noise"] + (True,)
            ch = os.path.join(noise2, "noise", "choice.json")
            s = json.load(open(ch))["sigma"] if os.path.exists(ch) else "?"
            desc["noise_big"] = f"шум с большим σ (выбран σ={s})"
    for v, d in VARIANTS:
        r = load_eval(os.path.join(run, "eval", v), test, "test")
        if r is not None:
            methods[v], desc[v] = r, d
    return test, methods, desc


def build(run, phase0, noise2):
    test, methods, desc = load_all(run, phase0, noise2)
    N = len(test)
    VER_KEYS = ("auroc_diag", "auroc_off", "ver_tpr_diag", "ver_tpr_off", "ver_fpr_diag", "ver_fpr_off",
                "vvote_cross", "vvote_self", "switch_acc", "switch_fixed", "switch_broken")
    S, V = {}, {}
    for k, v in methods.items():
        A, ok = v[0], v[3]
        if not ok:                                   # no verifier outputs: protocol-1 metrics unavailable
            A = dict(A, p_err=np.zeros_like(A["p_err"]))
        S[k], V[k] = summarize(A), per_problem(A)
        if not ok:
            for key in VER_KEYS:
                S[k][key] = float("nan")
            for key in ("vvote_cross", "vvote_self"):
                V[k][key] = V[k][key] * np.nan
    choice = json.load(open(os.path.join(run, "select", "choice.json"))) if os.path.exists(
        os.path.join(run, "select", "choice.json")) else {}
    L = ["# Фаза 1: обученные режимы на Qwen3 — отчёт\n"]
    st = os.path.join(run, "data", "stats.json")
    if os.path.exists(st):
        L.append(f"- данные: {json.load(open(st))}")
    if choice:
        L.append(f"- на этап 2 выбран **{choice['best']}** ({choice['why']}); val: "
                 + "; ".join(f"{k}: pass@1 {v['pass1']:.3f}, φ {fmt(v['phi'])}" for k, v in choice["val"].items()))
    state = os.path.join(run, "state.json")
    if os.path.exists(state):
        s = json.load(open(state))
        bad = {k: v["status"] for k, v in s.get("jobs", {}).items() if v.get("status") in ("failed", "skipped")}
        if bad:
            L.append(f"- **не выполнены:** {bad} (см. logs/)")

    # ---------------- verdicts
    def comp(X, R):
        Ax, Ar = methods[X][0], methods[R][0]
        wx, wr = ~Ax["correct"][:, 0], ~Ar["correct"][:, 0]
        op = Ax["op"]
        return {
            "d_pass": boot_ci(lambda i: V[X]["pass1"][i].mean() - V[R]["pass1"][i].mean(), N),
            "d_phi": boot_ci(lambda i: phi_within(wr, op, i) - phi_within(wx, op, i), N),
            "d_sw": boot_ci(lambda i: same_wrong(Ar, i) - same_wrong(Ax, i), N),
            "d_gauc": boot_ci(lambda i: graded_auroc(Ax, i) - graded_auroc(Ar, i), N),
            "d_maj": boot_ci(lambda i: V[X]["maj4"][i].mean() - V[R]["maj4"][i].mean(), N),
            "d_vote": boot_ci(lambda i: V[X]["vvote_cross"][i].mean() - V[R]["vvote_cross"][i].mean(), N),
        }

    rows = []
    best = choice.get("best")
    for X, refs in ((best, ["sftmulti", "p0_strategies", "p0_temperature", "noise_big"]),
                    ("s2_best", ["s2_ctrl", "p0_strategies", "p0_temperature", "noise_big"])):
        if X not in methods:
            continue
        for R in refs:
            if R not in methods:
                continue
            c = comp(X, R)
            ok = c["d_pass"][0] >= -TH["pass_margin"] and c["d_phi"][1] > 0 and (c["d_maj"][1] > 0 or c["d_vote"][1] > 0)
            rows.append((X, R, c, ok))
    L.append("\n## 1. Главный результат\n")
    L.append("Критерий (зафиксирован заранее): pass@1(X) ≥ pass@1(R) − 0.02, корреляция ошибок φ у X ниже "
             "(95% CI > 0), и итоговая точность (maj@4 или голосование с перекрёстной проверкой) выше (95% CI > 0). "
             "Все методы — на одних и тех же 1500 задачах теста фазы 0, интервалы — парный бутстрап.\n")
    L.append("| X | против R | Δ pass@1 | Δ φ ошибок (R − X) | Δ одинаковых неверных (R − X) | Δ AUROC несогласия | "
             "Δ maj@4 | Δ голосования с проверкой | успех |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for X, R, c, ok in rows:
        L.append(f"| {X} | {R} | {ci(c['d_pass'])} | {ci(c['d_phi'])} | {ci(c['d_sw'])} | {ci(c['d_gauc'])} | "
                 f"{ci(c['d_maj'])} | {ci(c['d_vote'])} | {'✅' if ok else '❌'} |")
    if not rows:
        L.append("Нет данных для сравнения (см. статус задач).")

    # ---------------- all methods
    L.append("\n## 2. Все методы на тесте\n")
    L.append("«Несогласие» — протокол 2 в градуированном виде: доля других режимов с другим ответом. "
             "≥2/3 и =1 — правила «тревога, если не согласны хотя бы 2 из 3 / все 3».\n")
    L.append("| метод | описание | режимов | pass@1 | maj@M | pass@M | φ ошибок | одинаковый неверный | ровно один прав | "
             "AUROC несогласия | несогласие ≥2/3: TPR / FPR | =1: TPR / FPR | голосование с проверкой |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, (A, bins, miss, _ok) in methods.items():
        s = S[k]
        e1 = float((A["correct"][:, 0].sum(0) == 1).mean())
        t23, f23 = graded_at(A, 2 / 3)
        t1, f1 = graded_at(A, 1.0)
        Mk = A["correct"].shape[0]
        L.append(f"| {k} | {desc[k]} | {Mk} | {fmt(s['pass1'])} | {fmt(s['maj4'])} | {fmt(s.get(f'pass@{Mk}'))} | "
                 f"{fmt(phi_within(~A['correct'][:, 0], A['op'], np.arange(N)))} | {fmt(same_wrong(A, np.arange(N)))} | "
                 f"{fmt(e1)} | {fmt(graded_auroc(A))} | {fmt(t23)} / {fmt(f23)} | {fmt(t1)} / {fmt(f1)} | "
                 f"{fmt(s['vvote_cross'])} |")
    L.append("\n## 3. Проверка Yes/No (протокол 1, обученный проверяющий у моделей фазы 1)\n")
    L.append("| метод | AUROC диаг. / вне | TPR@5%FPR диаг. / вне | сравнение со вторым сэмплом: TPR диаг. / вне | FPR диаг. / вне |")
    L.append("|---|---|---|---|---|")
    for k in methods:
        s = S[k]
        L.append(f"| {k} | {fmt(s['auroc_diag'])} / {fmt(s['auroc_off'])} | {fmt(s['ver_tpr_diag'])} / "
                 f"{fmt(s['ver_tpr_off'])} | {fmt(s['cmp_tpr_diag'])} / {fmt(s['cmp_tpr_off'])} | "
                 f"{fmt(s['cmp_fpr_diag'])} / {fmt(s['cmp_fpr_off'])} |")
    L.append("\nТочность по режимам (решение 0): " + "; ".join(
        f"{k} {[round(x, 3) for x in S[k]['acc_per_mode']]}" for k in methods))

    # ---------------- training / stage 2
    L.append("\n## 4. Обучение\n")
    L.append("| вариант | шагов сделано | эпох пройдено | токенов | часов | доля примеров по режимам |")
    L.append("|---|---|---|---|---|---|")
    for v, _ in VARIANTS[:6]:
        f = os.path.join(run, "train", v, "result.json")
        if os.path.exists(f):
            r = json.load(open(f))
            ep = r.get("epochs_done", 0.0)
            L.append(f"| {v} | {r['steps_done']} / лимит {r.get('time_limit_h', '?')} ч | {ep:.2f} | {r['tokens'] / 1e6:.1f}M | {r['hours']:.2f} | "
                     f"{[round(x, 2) for x in r['mode_share']] if r.get('mode_share') else '—'} |")
    for name in ("s2_best", "s2_ctrl"):
        f = os.path.join(run, "stage2", name, "stats.json")
        if os.path.exists(f):
            L.append(f"\nЭтап 2 {name}: " + "; ".join(
                f"раунд {s['round']}: ровно один режим прав {s['exactly_one_share']:.3f}, хоть один {s['any_share']:.3f}, "
                f"точность по режимам {[round(x, 3) for x in s['acc_per_mode']]}" for s in json.load(open(f))))
    L.append("\n## Как читать\n")
    L.append("- Раздел 1: ✅ против `sftmulti` / `s2_ctrl` — эффект от режимов, а не от данных; ✅ против "
             "`p0_strategies` — обученные режимы лучше промптов; против `noise_big` — лучше шума.\n"
             "- «Δ φ» > 0 при ❌ в итоговой точности: ошибки развелись, но голосованию это пока не помогло.\n"
             "- Раздел 3: у обученных моделей проверяющий учился на естественных ошибках; сравните AUROC с фазой 0 (≈0.62).\n"
             "- Раздел 4: доля примеров по режимам показывает, не «умер» ли какой-то режим.")
    out = os.path.join(run, "report_phase1.md")
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    json.dump({"summary": S, "verdicts": [(X, R, c, ok) for X, R, c, ok in rows], "choice": choice},
              open(os.path.join(run, "summary_phase1.json"), "w"), default=float, indent=1)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--phase0", required=True)
    ap.add_argument("--noise2", default="")
    ap.add_argument("--mark", default="")
    a = ap.parse_args()
    print(build(a.run, a.phase0, a.noise2))
    if a.mark:
        os.makedirs(a.mark, exist_ok=True)
        open(os.path.join(a.mark, "DONE"), "w").close()
