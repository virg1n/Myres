"""Turn results/summary.json into results/results.md (+ a plot if matplotlib
is installed).  python -m revise.report --config configs/small.yaml"""

import argparse
import json
import os

from .utils import load_config

LABELS = {
    "plain": "1. plain model, no revision",
    "fix_only": "2. revision, no erasing",
    "erase_suffix": "3. erase all text after the error",
    "dep_head": "4. erase dependent tokens (dep head)  [ours]",
    "oracle_dep": "5. erase true dependents (oracle)",
    "dep_head_lines": "6. ours, erase whole lines",
    "random_erase": "7. random erasure, same amount as 4",
    "confdrop_measure": "4b. ours, measured confidence drop",
    "revision_model_no_revision": "(diag) revision model, revision off",
}


def label(k):
    base, *suf = k.split("@")
    name = LABELS.get(base, base)
    for x in suf:
        name += " [oracle detection]" if x == "oracledetect" else f" [{x}]"
    return name


def _order(rows):
    keys = [k for k in LABELS if k in rows]
    keys += [k + "@oracledetect" for k in LABELS if k + "@oracledetect" in rows]
    return keys + sorted(k for k in rows if k not in keys)


def _pct(x):
    return "-" if x is None else f"{100 * x:.1f}"


def _f(x, d=2):
    return "-" if x is None else f"{x:.{d}f}"


def write_report(cfg):
    out_dir = os.path.join(cfg["out_dir"], "results")
    with open(os.path.join(out_dir, "summary.json")) as f:
        summary = json.load(f)
    lines = [f"# Results: {cfg['name']}", ""]
    lines.append(f"Model: d_model={cfg['model']['d_model']}, layers={cfg['model']['n_layers']}; "
                 f"window={cfg['revision']['window']} tokens; dep target={cfg['revision']['dep_target']}; "
                 f"max fixes/problem={cfg['eval']['policy']['max_fixes']}.")
    summary = dict(summary)
    if cfg["revision"].get("frozen_writer"):
        lines.append("Frozen writer: every variant writes with the same plain model; a separately trained reviser "
                     "(initialised from it) detects, fixes, erases and fills. Variants differ only in revision.")
    else:
        lines.append("All revision variants share one model (trained at the same FLOPs as the plain model); they "
                     "differ only in what is erased after a fix.")
    lines.append("FLOPs are per problem, relative to plain greedy decoding, counted as if the unchanged prefix were "
                 "KV-cached. *[oracle detection]*: the checker finds the true first error and it is fixed with the "
                 f"correct token {cfg['eval'].get('oracle_delay', 24)} tokens after it was written; the erasure "
                 "policy then decides what else to erase. This isolates erasure from error detection.")
    calib = summary.pop("calibration", None)
    if calib:
        lines.append(f"Error threshold calibrated on {calib['n_pos_windows'] + calib['n_neg_windows']} windows of the "
                     f"model's own held-out solutions: tau_err={calib['tau_err']:.3f} (allows "
                     f"{calib['false_per_problem']} false fixes per solution); detection recall at that threshold "
                     f"{100 * calib['detect_recall']:.1f}%.")
    lines.append("")
    for set_name, rows in summary.items():
        ops = {"id": cfg["data"]["op_test_id"], "ood": cfg["data"]["op_test_ood"]}[set_name]
        lines.append(f"## Test set: {set_name.upper()} (op {ops[0]}-{ops[1]}, n={next(iter(rows.values()))['n']})")
        lines.append("")
        lines.append("| variant | acc % | ± | fully correct % | FLOPs ×plain | passes | fixes | lag | erased/fix | "
                     "fix prec % | erase prec % | erase rec % |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        order = _order(rows)
        for k in order:
            s = rows[k]
            lines.append(
                f"| {label(k)} | {_pct(s['acc'])} | {_pct(s['acc_se'])} | {_pct(s['sol_correct'])} | "
                f"{_f(s.get('flops_x_plain'))} | {_f(s['steps'], 0)} | {_f(s['fixes'])} | {_f(s.get('lag_per_fix'), 1)} | "
                f"{_f(s['erased_per_fix'], 1)} | {_pct(s.get('fix_precision'))} | "
                f"{_pct(s.get('erase_precision_vs_true_deps'))} | {_pct(s.get('erase_recall_vs_true_deps'))} |")
        lines.append("")
        lines.append("Accuracy by number of operations:")
        lines.append("")
        all_ops = sorted({int(o) for s in rows.values() for o in s["acc_by_op"]})
        lines.append("| variant | " + " | ".join(str(o) for o in all_ops) + " |")
        lines.append("|---|" + "---|" * len(all_ops))
        for k in order:
            if "@maxfix" in k:
                continue
            a = rows[k]["acc_by_op"]
            lines.append(f"| {label(k)} | " + " | ".join(_pct(a.get(str(o))) for o in all_ops) + " |")
        lines.append("")
    lines += [
        "Columns: *lag* = tokens written after the error before it was fixed; "
        "*fix prec* = share of fixes made at a token the checker labels as a root error; "
        "*erase prec / rec* = overlap of the erased tokens with the true dependents of the fixed token "
        "(generator dependency graph).",
        "",
    ]
    path = os.path.join(out_dir, "results.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {path}")
    _plot(summary, out_dir)


def _plot(summary, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, len(summary), figsize=(6 * len(summary), 4.5), squeeze=False)
    for ax, (set_name, rows) in zip(axes[0], summary.items()):
        groups = {}
        for k, s in rows.items():
            base = k.split("@maxfix")[0]
            if s.get("flops_x_plain") is None:
                continue
            groups.setdefault(base, []).append((s["flops_x_plain"], s["acc"]))
        order = [k for k in LABELS if k in groups] + sorted(k for k in groups if k not in LABELS)
        for base in order:
            pts = groups[base]
            pts.sort()
            ax.plot([p[0] for p in pts], [100 * p[1] for p in pts], marker="o", label=label(base))
        ax.set_xscale("log")
        ax.set_xlabel("inference FLOPs (× plain greedy)")
        ax.set_ylabel("accuracy %")
        ax.set_title(f"iGSM {set_name.upper()}")
        ax.grid(alpha=0.3)
    axes[0][-1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "acc_vs_compute.png"), dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    write_report(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
