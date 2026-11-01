"""Extra, harder training problems for phase 1 (the phase-0 pool was mostly solved 8/8).

  - phase-0 pool problems solved <= 6 of 8 times (re-sampled: 16 solutions in total)
  - MATH train level 5 with numeric answers
  - synthetic train-template problems with 6-10 operations
Writes <out>/hard/pool.jsonl; the GPU jobs (phase0.gen pool) solve it with 4 strategies x 2."""
import argparse
import gzip
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase0 import data as D  # noqa: E402
from phase0 import synth  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase0", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_math", type=int, default=1500)
    ap.add_argument("--n_synth", type=int, default=2000)
    ap.add_argument("--max_resample", type=int, default=2500)
    a = ap.parse_args()
    hd = os.path.join(a.out, "hard")
    os.makedirs(hd, exist_ok=True)
    pool = {p["id"]: p for p in D.load_jsonl(os.path.join(a.phase0, "data", "pool.jsonl"))}
    k = {}
    pd = os.path.join(a.phase0, "pool")
    for f in sorted(os.listdir(pd)):
        if f.endswith(".jsonl.gz"):
            with gzip.open(os.path.join(pd, f), "rt") as fh:
                for l in fh:
                    r = json.loads(l)
                    c = k.setdefault(r["id"], [0, 0])
                    c[0] += sum(r["correct"])
                    c[1] += len(r["correct"])
    rows = [pool[i] for i, (ok, n) in k.items() if i in pool and ok <= n - 2][:a.max_resample]
    notes = []
    m5 = D.load_math("train", (5,), notes)[:a.n_math] if a.n_math > 0 else []
    rows += m5
    sy = synth.make_set(7, "train", a.n_synth, 6, 10)
    for i, r in enumerate(sy):
        r.update(id=f"synth-hard-{i}", source="synth", bin=f"synth_{r.pop('ops')}")
    rows += sy
    D.save_jsonl(os.path.join(hd, "pool.jsonl"), rows)
    shutil.copy(os.path.join(a.phase0, "data", "test.jsonl"), os.path.join(hd, "test.jsonl"))   # gen.py loads it
    print(f"hard pool: {len(rows)} problems (resampled {len(rows) - len(m5) - len(sy)}, math L5 {len(m5)}, "
          f"synth {len(sy)}) {notes}", flush=True)
    open(os.path.join(hd, "DONE"), "w").close()


if __name__ == "__main__":
    main()
