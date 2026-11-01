"""Choose the stage-1 variant that goes to stage 2 -- on val, by a rule fixed in advance:
among mode variants with pass@1 >= pass@1(sftmulti) - 0.02, the lowest error correlation (phi within
difficulty); if none qualifies, the mode variant with the highest pass@1."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase1.metrics import load_eval, val_summary  # noqa: E402

MODE_VARIANTS = ["filmmean", "wta1", "wtamulti", "wtabal"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--margin", type=float, default=0.02)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(os.path.join(a.run, "data", "val.jsonl"))]
    summ = {}
    for v in ["sft1", "sftmulti"] + MODE_VARIANTS:
        r = load_eval(os.path.join(a.run, "eval", v), rows, "val", with_verify=False)
        if r is not None:
            summ[v] = val_summary(r[0])
    if "sftmulti" not in summ:
        raise SystemExit("sftmulti evaluation is missing: the pass@1 constraint cannot be checked")
    base = summ["sftmulti"]["pass1"]
    cand = [v for v in MODE_VARIANTS if v in summ and summ[v]["pass1"] >= base - a.margin]
    if cand:
        best = min(cand, key=lambda v: summ[v]["phi"] if summ[v]["phi"] == summ[v]["phi"] else 9)
        why = "наименьшая корреляция ошибок среди вариантов с допустимым pass@1"
    else:
        pool = [v for v in MODE_VARIANTS if v in summ]
        if not pool:
            raise SystemExit("no evaluated mode variant")
        best = max(pool, key=lambda v: summ[v]["pass1"])
        why = "ни один вариант не прошёл по pass@1 — взят вариант с наибольшим pass@1"
    out = {"best": best, "why": why, "val": summ, "adapter": os.path.join(a.run, "train", best, "adapter.pt")}
    os.makedirs(os.path.join(a.run, "select"), exist_ok=True)
    json.dump(out, open(os.path.join(a.run, "select", "choice.json"), "w"), indent=1)
    open(os.path.join(a.run, "select", "DONE"), "w").close()
    print("selected", best, "-", why, summ, flush=True)


if __name__ == "__main__":
    main()
