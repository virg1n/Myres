"""Training data for ModC: the same train problems, with the strategy that produced each reference.

The pool was generated with 4 system-prompt strategies, so the strategy plays the role of the teacher
in the ModC paper. For every train problem: the first correct solution of every strategy (<= 4).
No generation; reads the pool files only."""
import argparse
import gzip
import json
import os


def read_gz(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(l) for l in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir with train.jsonl of phase 1")
    ap.add_argument("--pools", nargs="+", required=True, help="dirs with pool_*.jsonl.gz")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.data, "train.jsonl")) as f:
        train = [json.loads(l) for l in f if l.strip()]
    ids = {r["id"] for r in train}
    per = {}
    for d in a.pools:
        for name in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if name.startswith("pool_") and name.endswith(".jsonl.gz"):
                for r in read_gz(os.path.join(d, name)):
                    if r["id"] in ids and isinstance(r.get("mode"), int):
                        for t, c in zip(r["texts"], r["correct"]):
                            if c:
                                per.setdefault(r["id"], {}).setdefault(r["mode"], t)
    rows, hist, by_mode = [], [0] * 5, [0] * 4
    for r in train:
        refs = per.get(r["id"], {})
        if not refs:
            continue
        # these texts are not the refs of train.jsonl: their control is the variant sftmodc
        rows.append({"id": r["id"], "source": r["source"], "question": r["question"], "answer": r["answer"],
                     "refs": [{"mode": k, "text": refs[k]} for k in sorted(refs)]})
        hist[len(refs)] += 1
        for k in refs:
            by_mode[k] += 1
    if len(rows) != len(train):       # train.jsonl was built from the same pools: a gap means wrong input
        raise SystemExit(f"labelled references for {len(rows)} of {len(train)} train problems in {a.pools}")
    with open(os.path.join(a.out, "train.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"train_problems": len(train), "with_refs": len(rows), "refs_hist": hist, "refs_by_mode": by_mode}
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    open(os.path.join(a.out, "DONE"), "w").close()
    print("modc data:", stats, flush=True)


if __name__ == "__main__":
    main()
