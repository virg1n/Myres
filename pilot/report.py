"""Build report.md (+ summary.json, plots) from all finished evaluations of a pilot run."""
import argparse
import json
import os
import time

import numpy as np

from .common import read_json, write_json
from .metrics import per_problem

METHODS = [
    ("A", "baseline (T токенов)"),
    ("Aprime", "baseline, равные FLOPs с WTA"),
    ("B_fix", "A + 4 фиксированных вектора шума в весах"),
    ("B_cont", "A + свежий шум в весах на каждую генерацию"),
    ("C_fix", "4 кода + relaxed WTA (этап 1)"),
    ("C_cont", "непрерывный z + relaxed WTA (этап 1)"),
    ("D_fix", "4 кода, обычный loss (контроль)"),
    ("D_cont", "непрерывный z, обычный loss (контроль)"),
    ("A_RFT", "A + этап 2 без режимов (контроль)"),
    ("C_fix2", "C_fix + этап 2, бонус 1/|W|"),
    ("C_fix2u", "C_fix + этап 2, равные веса (контроль бонуса)"),
    ("C_cont2", "C_cont + этап 2, бонус 1/|W|"),
]
# candidate -> reference baseline (same budget)
CANDIDATES = [("C_fix", "Aprime"), ("C_cont", "Aprime"), ("C_fix2", "A_RFT"), ("C_cont2", "A_RFT"),
              ("B_fix", "A"), ("B_cont", "A")]
SETS = [("id", "ID (сложность как в обучении)"), ("ood", "OOD (сложнее обучения)")]


def load(run):
    res = {}
    for m, _ in METHODS:
        d = os.path.join(run, "eval", m)
        if os.path.exists(os.path.join(d, "metrics.json")):
            res[m] = {"dir": d, "info": read_json(os.path.join(d, "metrics.json"))}
    return res


def vectors(res, m, s, cache={}):
    key = (res[m]["dir"], s)
    if key not in cache:
        A = dict(np.load(os.path.join(res[m]["dir"], f"arrays_{s}.npz")))
        cache[key] = per_problem(A)
    return cache[key]


def ratio_stat(v, base, idx):
    num, den = v[base + "_tp"], v[base + "_pos"]
    return num[idx].sum(-1) / np.maximum(den[idx].sum(-1), 1)


def boot(fn, N, R=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, N, size=(R, N))
    point = fn(np.arange(N)[None])[0]
    samples = fn(idx)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def fmt(x, d=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{d}f}"


def ci(t):
    return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def verdicts(res):
    rows = []
    for cand, ref in CANDIDATES:
        if cand not in res or ref not in res:
            continue
        for s, _ in SETS:
            vc, vr = vectors(res, cand, s), vectors(res, ref, s)
            N = len(vc["pass1"])
            d_pass = boot(lambda i: vc["pass1"][i].mean(-1) - vr["pass1"][i].mean(-1), N)
            d_cross = boot(lambda i: ratio_stat(vc, "ver_off", i) - ratio_stat(vc, "ver_diag", i), N)
            d_ref = boot(lambda i: ratio_stat(vc, "ver_off", i) - ratio_stat(vr, "ver_diag", i), N)
            d_e2e = boot(lambda i: vc["vvote_cross"][i].mean(-1) - vr["vvote_self"][i].mean(-1), N)
            d_maj = boot(lambda i: vc["vvote_cross"][i].mean(-1) - vr["maj4"][i].mean(-1), N)
            ok = d_pass[0] >= -0.03 and d_cross[1] > 0 and d_e2e[1] > 0
            rows.append({"cand": cand, "ref": ref, "set": s, "d_pass1": d_pass, "d_cross_minus_self": d_cross,
                         "d_tpr_vs_ref_self": d_ref, "d_e2e_vs_ref_selfvote": d_e2e,
                         "d_e2e_vs_ref_maj4": d_maj, "success": ok})
    return rows


def plots(res, run):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    out = []
    for s, title in SETS:
        ms = [m for m, _ in METHODS if m in res and s in res[m]["info"]["metrics"]]
        if not ms:
            continue
        cols = 4
        rows = (len(ms) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, m in zip(axes.flat, ms):
            mat = np.array(res[m]["info"]["metrics"][s]["tpr_matrix"], dtype=float)
            ax.axis("on")
            im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis")
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    ax.text(j, i, fmt(mat[i, j], 2), ha="center", va="center", color="w", fontsize=8)
            ax.set_title(m, fontsize=9)
            ax.set_xlabel("проверяющий j", fontsize=8)
            ax.set_ylabel("решающий i", fontsize=8)
        fig.suptitle(f"TPR при FPR≈5%: кто кого проверяет — {title}", fontsize=10)
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
        p = os.path.join(run, f"R_matrix_{s}.png")
        fig.savefig(p, dpi=110, bbox_inches="tight")
        plt.close(fig)
        out.append(os.path.basename(p))
        fig, ax = plt.subplots(figsize=(6, 4))
        ks = [1, 2, 4, 8, 16]
        for m in ms:
            mt = res[m]["info"]["metrics"][s]
            ys = [mt["pass1"]] + [mt.get(f"pass@{k}") for k in ks[1:]]
            ax.plot(ks, ys, marker="o", label=m)
            if "fresh_pass@k" in mt:
                ax.plot(ks, [mt["fresh_pass@k"][str(k)] if str(k) in mt["fresh_pass@k"] else mt["fresh_pass@k"][k]
                             for k in ks], ls="--", label=m + " (свежие z)")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("k")
        ax.set_ylabel("pass@k")
        ax.set_title(title)
        ax.legend(fontsize=7)
        p = os.path.join(run, f"pass_at_k_{s}.png")
        fig.savefig(p, dpi=110, bbox_inches="tight")
        plt.close(fig)
        out.append(os.path.basename(p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--mark", default="", help="directory to put a DONE marker into (orchestrator)")
    a = ap.parse_args()
    run = a.run
    res = load(run)
    state = read_json(os.path.join(run, "state.json")) if os.path.exists(os.path.join(run, "state.json")) else {}
    L = [f"# Пилот seed-режимов на iGSM — отчёт\n", f"Сгенерирован: {time.strftime('%Y-%m-%d %H:%M')}\n"]
    plan = state.get("plan", {})
    if plan:
        L.append(f"- op_max (по калибровке): **{plan.get('op_max')}**; токенов на модель этапа 1: "
                 f"**{plan.get('tokens', 0) / 1e9:.2f}B** (A′: {plan.get('tokens_aprime', 0) / 1e9:.2f}B); "
                 f"задач на этапе 2: {plan.get('stage2_problems')}")
        L.append(f"- Пропускная способность: baseline {plan.get('tps_base', 0) / 1e3:.0f}k ток/с, "
                 f"WTA {plan.get('tps_wta', 0) / 1e3:.0f}k ток/с, генерация {plan.get('gps', 0):.0f} решений/с")
    for w in state.get("warnings", []):
        L.append(f"\n**Предупреждение:** {w}")
    jobs = state.get("jobs", {})
    bad = {k: v for k, v in jobs.items() if v.get("status") in ("failed", "skipped")}
    if bad:
        L.append("\n**Неудачные/пропущенные задачи:** " + ", ".join(f"{k} ({v['status']})" for k, v in bad.items())
                 + " — логи в `logs/`.")
    missing = [m for m, _ in METHODS if m not in res]
    if missing:
        L.append(f"\nНет результатов для: {', '.join(missing)}")

    V = verdicts(res)
    L.append("\n## 1. Главный результат: заранее зафиксированный критерий\n")
    L.append("Успех кандидата X против базы R (тот же бюджет): (а) pass@1(X) ≥ pass@1(R) − 0.03; "
             "(б) смена z помогает проверке: TPR@5%FPR вне диагонали − на диагонали > 0 (95% CI); "
             "(в) итоговая точность «голосование, взвешенное перекрёстной проверкой» у X выше, чем "
             "«голосование, взвешенное самопроверкой» у R (95% CI). Ошибки естественные, CI — парный бутстрап по задачам.\n")
    L.append("| X | R | набор | Δpass@1 | (б) TPR вне диаг. − диаг. | TPR X(вне диаг.) − R(самопроверка) | "
             "(в) Δ итоговой точности vs R self-vote | Δ vs R maj@4 | успех |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in V:
        L.append(f"| {r['cand']} | {r['ref']} | {r['set']} | {ci(r['d_pass1'])} | {ci(r['d_cross_minus_self'])} | "
                 f"{ci(r['d_tpr_vs_ref_self'])} | {ci(r['d_e2e_vs_ref_selfvote'])} | {ci(r['d_e2e_vs_ref_maj4'])} | "
                 f"{'✅' if r['success'] else '❌'} |")
    L.append("\nДля B (шум без обучения) критерий тот же, база — A. Колонка «TPR X − R» сравнивает разные наборы "
             "ошибок (у каждой модели свои), поэтому это вспомогательная величина.")

    for s, title in SETS:
        L.append(f"\n## 2. Качество решения — {title}\n")
        L.append("| метод | описание | τ | pass@1 (среднее по режимам) | maj@4 | pass@4 | pass@16 | pass@16 свежие z |")
        L.append("|---|---|---|---|---|---|---|---|")
        for m, desc in METHODS:
            if m not in res or s not in res[m]["info"]["metrics"]:
                continue
            info, mt = res[m]["info"], res[m]["info"]["metrics"][s]
            fz = mt.get("fresh_pass@k", {})
            L.append(f"| {m} | {desc} | {info.get('temp')}{' σ=' + str(info['sigma']) if 'sigma' in info else ''} | "
                     f"{fmt(mt['pass1'])} | {fmt(mt['maj4'])} | {fmt(mt.get('pass@4'))} | {fmt(mt.get('pass@16'))} | "
                     f"{fmt(fz.get('16')) if fz else '—'} |")
        L.append(f"\n## 3. Проверка ошибок — {title}\n")
        L.append("Протокол 1 — обученный проверяющий читает решение. Протокол 2 — проверяющий решает сам и сравнивает "
                 "ответ. «Диаг.» — проверяет тот же режим (в свежем контексте), «вне» — другой режим.\n")
        L.append("| метод | ошибок (режим 0) | AUROC диаг. / вне | TPR@5%FPR диаг. / вне | FPR факт. | "
                 "локализация диаг. / вне | сравнение: TPR диаг. / вне | сравнение: FPR диаг. / вне |")
        L.append("|---|---|---|---|---|---|---|---|")
        for m, _ in METHODS:
            if m not in res or s not in res[m]["info"]["metrics"]:
                continue
            mt = res[m]["info"]["metrics"][s]
            L.append(f"| {m} | {mt['n_errors_mode0']} | {fmt(mt['auroc_diag'])} / {fmt(mt['auroc_off'])} | "
                     f"{fmt(mt['ver_tpr_diag'])} / {fmt(mt['ver_tpr_off'])} | {fmt(mt['ver_fpr_off'])} | "
                     f"{fmt(mt['ver_loc_diag'])} / {fmt(mt['ver_loc_off'])} | {fmt(mt['cmp_tpr_diag'])} / "
                     f"{fmt(mt['cmp_tpr_off'])} | {fmt(mt['cmp_fpr_diag'])} / {fmt(mt['cmp_fpr_off'])} |")
        L.append(f"\n## 4. Итоговая точность процедур (4 решения + проверка) — {title}\n")
        L.append("| метод | одно решение | maj@4 | голосование + самопроверка | голосование + перекрёстная проверка | "
                 "замена при тревоге: точность (+исправлено / −испорчено) |")
        L.append("|---|---|---|---|---|---|")
        for m, _ in METHODS:
            if m not in res or s not in res[m]["info"]["metrics"]:
                continue
            mt = res[m]["info"]["metrics"][s]
            L.append(f"| {m} | {fmt(mt['pass1'])} | {fmt(mt['maj4'])} | {fmt(mt['vvote_self'])} | "
                     f"{fmt(mt['vvote_cross'])} | {fmt(mt['switch_acc'])} (+{fmt(mt['switch_fixed'])} / "
                     f"−{fmt(mt['switch_broken'])}) |")
        L.append(f"\n## 5. Диагностика режимов — {title}\n")
        L.append("| метод | φ ошибок внутри сложности | одинаковый первый ошибочный параметр | режимы дают разные ответы | "
                 "разные решения | доля обратного стиля по режимам | точность по режимам |")
        L.append("|---|---|---|---|---|---|---|")
        for m, _ in METHODS:
            if m not in res or s not in res[m]["info"]["metrics"]:
                continue
            mt = res[m]["info"]["metrics"][s]
            L.append(f"| {m} | {fmt(mt['err_phi_within_op'])} | {fmt(mt['same_first_error_param'])} | "
                     f"{fmt(mt['modes_differ_answer'])} | {fmt(mt['modes_differ_solution'])} | "
                     f"{[round(x, 2) for x in mt['backward_share_per_mode']]} | "
                     f"{[round(x, 3) for x in mt['acc_per_mode']]} |")
    L.append("\n## 6. Обучение\n")
    L.append("| прогон | токенов | ток/с | val acc по режимам (greedy) | доля назначений WTA по кодам |")
    L.append("|---|---|---|---|---|")
    for name in sorted(os.listdir(os.path.join(run, "train"))) if os.path.isdir(os.path.join(run, "train")) else []:
        p = os.path.join(run, "train", name, "result.json")
        if os.path.exists(p):
            r = read_json(p)
            L.append(f"| {name} | {r.get('tokens', 0) / 1e9:.2f}B | {fmt(r.get('tok_per_s'), 0)} | "
                     f"{[round(x, 3) for x in r.get('val_acc_per_mode', [])]} | "
                     f"{[round(x, 3) for x in r['win_share']] if r.get('win_share') else '—'} |")
    imgs = plots(res, run)
    if imgs:
        L.append("\n## 7. Графики\n")
        for p in imgs:
            L.append(f"![{p}]({p})")
    L.append("\n## Как читать\n")
    L.append("- **Главная гипотеза подтверждается**, если у C_fix2 / C_cont2 в разделе 1 стоит ✅ хотя бы на одном "
             "наборе, и при этом у A_RFT нет такого же эффекта (у baseline диагональ = вне диагонали по построению).\n"
             "- C ≈ D по разделу 5 (режимы не различаются) → WTA не создал режимы.\n"
             "- Режимы есть (разные ответы/стили), но (б) ≈ 0 → разнообразие не переносится на проверку.\n"
             "- Протокол «сравнение» ≈ обученный проверяющий → ценность в разнообразии решений, а не в проверке.\n"
             "- C_fix2 ≈ C_fix2u → бонус 1/|W| не важен. B ≈ C → хватает шума без обучения.\n"
             "- cont vs fix: сравните pass@16 «свежие z» с pass@16 фиксированных кодов.")
    with open(os.path.join(run, "report.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    write_json(os.path.join(run, "summary.json"),
               {"verdicts": V, "metrics": {m: r["info"]["metrics"] for m, r in res.items()}, "plan": plan})
    print(os.path.join(run, "report.md"))
    if a.mark:
        os.makedirs(a.mark, exist_ok=True)
        open(os.path.join(a.mark, "DONE"), "w").close()


if __name__ == "__main__":
    main()
