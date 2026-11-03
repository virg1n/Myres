"""Stage 2: WTA by correctness of self-generated solutions, in offline rounds.

Each round: export the current adapter's modes -> every mode solves the stage-2 problems once
(model without modes: 4 samples) -> winners W = modes with the correct answer ->
train on the winners' own solutions with weight 1/|W| (a mode that alone solves a problem gets the
full weight; for the control without modes each correct sample gets 1/4, i.e. plain RFT).
Logged per round: share of problems solved by exactly one mode, by any mode, per-mode accuracy."""
import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase1.evaluate import adapter_cfg, export, gen, n_modes  # noqa: E402

PY = sys.executable


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def read_gz(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(l) for l in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--choice", default="", help="select.json: use the adapter of the chosen variant")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--round_hours", type=float, default=0.5, help="training time limit per round")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_tokens", type=int, default=1200)
    ap.add_argument("--max_steps", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if os.path.exists(os.path.join(a.out, "DONE")):
        return
    adapter = a.adapter
    if a.choice:
        ch = json.load(open(a.choice))
        adapter = ch["adapter"]
        json.dump(ch, open(os.path.join(a.out, "init_choice.json"), "w"))
    K = n_modes(adapter)
    probs = {}
    with open(os.path.join(a.data, "stage2.jsonl")) as f:
        for l in f:
            r = json.loads(l)
            probs[r["id"]] = r
    stats = []
    for rd in range(a.rounds):
        rdir = os.path.join(a.out, f"round{rd}")
        os.makedirs(rdir, exist_ok=True)
        if not os.path.exists(os.path.join(rdir, "train", "DONE")):
            tmp = os.path.join(rdir, "tmp_export")
            try:
                if K == 0:
                    if not os.path.exists(os.path.join(rdir, "solve_0_stage2.jsonl.gz")):
                        d = export(a.base, adapter, tmp, [None])[0]
                        gen("mode_solve", d, a.data, rdir, ["--tag", "0", "--sets", "stage2", "--n", "4",
                                                            "--seed", str(100 * (rd + 1))],
                            max_tokens=a.max_tokens)
                else:
                    todo = [k for k in range(K) if not os.path.exists(os.path.join(rdir, f"solve_{k}_stage2.jsonl.gz"))]
                    if todo:
                        dirs = export(a.base, adapter, tmp, todo)
                        for k, d in zip(todo, dirs):
                            gen("mode_solve", d, a.data, rdir, ["--tag", str(k), "--sets", "stage2", "--n", "1",
                                                                "--seed", str(100 * (rd + 1) + k + 1)],
                                max_tokens=a.max_tokens)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            rows, st = winners(rdir, K, probs)
            st["round"] = rd
            with open(os.path.join(rdir, "winners.jsonl"), "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            json.dump(st, open(os.path.join(rdir, "stats.json"), "w"), indent=1)
            log(f"round {rd}: {st}")
            cmd = [PY, "-m", "phase1.train", "--variant", "fixed", "--base", a.base, "--data", a.data,
                   "--train_file", os.path.join(rdir, "winners.jsonl"), "--init", adapter,
                   "--out", os.path.join(rdir, "train"), "--lr", str(a.lr), "--max_hours", str(a.round_hours),
                   "--film_modes", str(adapter_cfg(adapter).get("film_modes", 0)), "--seed", str(rd + 1)] + (["--max_steps", str(a.max_steps)] if a.max_steps else [])
            if subprocess.call(cmd) != 0:
                raise SystemExit("stage-2 training failed")
        stats.append(json.load(open(os.path.join(rdir, "stats.json"))))
        adapter = os.path.join(rdir, "train", "adapter.pt")
    shutil.copy(adapter, os.path.join(a.out, "adapter.pt"))
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    open(os.path.join(a.out, "DONE"), "w").close()
    log("stage 2 done", stats)


def winners(rdir, K, probs):
    if K == 0:
        sols = {r["id"]: r for r in read_gz(os.path.join(rdir, "solve_0_stage2.jsonl.gz"))}
        per = {i: [(t, c, None) for t, c in zip(r["texts"], r["correct"])] for i, r in sols.items()}
        n_slots = 4
    else:
        per = {}
        for k in range(K):
            for r in read_gz(os.path.join(rdir, f"solve_{k}_stage2.jsonl.gz")):
                per.setdefault(r["id"], []).append((r["texts"][0], r["correct"][0], k))
        n_slots = K
    rows, n_exact1, n_any, acc = [], 0, 0, [0.0] * max(K, 1)
    for i, lst in per.items():
        W = [(t, m) for t, c, m in lst if c]
        n_any += bool(W)
        n_exact1 += len(W) == 1
        for t, c, m in lst:
            acc[m if m is not None else 0] += c
        for t, m in W:
            w = 1.0 / len(W) if K else 1.0 / n_slots
            rows.append({"question": probs[i]["question"], "solution": t, "mode": m, "w": w})
    n = max(1, len(per))
    st = {"problems": len(per), "exactly_one_share": n_exact1 / n, "any_share": n_any / n,
          "winners": len(rows), "acc_per_mode": [x / (n * (4 if K == 0 else 1)) for x in acc]}
    return rows, st


if __name__ == "__main__":
    main()
