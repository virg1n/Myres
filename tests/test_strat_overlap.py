"""phase1.strat_overlap on fabricated phase-0 outputs (CPU, seconds).

    PYTHONUTF8=1 python3 -m tests.test_strat_overlap

1. estimators on hand-checkable inputs (unbiased miss probability, same-wrong pairs)
2. identical strategies: no gain from switching strategy; strategies good on disjoint problems: a large gain
3. the whole script on fake runs/phase0 + runs/phase1 directories: report and json written"""
import gzip
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile

from phase1 import strat_overlap as SO


def close(a, b, tol=1e-9):
    assert abs(a - b) <= tol, (a, b)


def test_estimators():
    close(SO.miss(2, 1, 1), 0.5)
    close(SO.miss(2, 1, 2), 0.0)
    close(SO.miss(4, 1, 2), 0.5)
    close(SO.miss(8, 0, 8), 1.0)
    assert SO.pairs_same([("3", False), ("3", False), ("4", False), ("", False)]) == (3, 1)
    assert SO.pairs_same([("3", False), ("5", True)], [("3", False), ("", False)]) == (1, 1)
    d = {0: [("1", True), ("2", False)], 1: [("1", True), ("1", True)], 2: [("2", False), ("2", False)],
         3: [("3", False), ("1", True)]}
    s = SO.problem_stats(d)
    # p = .5, 1, 0, .5; same strategy pass@2: 1, 1, 0, 1 -> .75; different: mean over 6 pairs of 1-(1-pa)(1-pb)
    close(s["p2_same"], 0.75)
    close(s["p2_diff"], (0.75 + 1 + 0.5 + 1 + 1 + 0.5) / 6)
    close(s["p8_mix"], 1.0)
    close(s["p4_mix"], 1 - 0.5 * 0 * 1 * 0.5)
    assert s["solved_by"] == 3
    assert s["sw_same"] == (1, 1)                 # the pair inside strategy 2 ("2", "2")
    assert SO.problem_stats({0: d[0], 1: d[1], 2: d[2], 3: d[3][:1]}) is None


def fake(rng, n_prob, mode_p, n=2, neutral=False):
    """per[id][mode] with success prob mode_p(problem, mode); wrong answers from a small set."""
    per = {}
    for i in range(n_prob):
        modes = list(range(4)) + (["neutral"] if neutral else [])
        for m in modes:
            k = 8 if m == "neutral" else n
            p = mode_p(i, m)
            per.setdefault(f"q{i}", {})[m] = [("0", True) if rng.random() < p else (str(rng.randint(1, 3)), False)
                                              for _ in range(k)]
    return per


def test_gain():
    rng = random.Random(0)
    same = fake(rng, 3000, lambda i, m: [0.1, 0.5, 0.9][i % 3], neutral=True)
    s = SO.by_source(same, sorted(same), {})["all"]
    assert abs(s["gain_p2"][0]) < 0.015 and s["gain_p2"][1] < 0 < s["gain_p2"][2], s["gain_p2"]
    assert abs(s["gain_p8"][0]) < 0.02, s["gain_p8"]
    disj = fake(rng, 3000, lambda i, m: 0.9 if m == "neutral" or m == i % 4 else 0.05, neutral=True)
    s = SO.by_source(disj, sorted(disj), {})["all"]
    # same strategy: mean over modes of 1-(1-p)^2 = (0.99 + 3 * 0.0975) / 4; different: 3 pairs of 6 involve the good one
    want = (3 * (1 - 0.1 * 0.95) + 3 * (1 - 0.95 ** 2)) / 6 - (0.99 + 3 * (1 - 0.95 ** 2)) / 4
    assert abs(s["gain_p2"][0] - want) < 0.02 and s["gain_p2"][1] > 0.02, (s["gain_p2"], want)
    assert "заметно" in SO.verdict(s["gain_p2"])


def write_gz(path, rows):
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_script(d):
    rng = random.Random(1)
    p0, p1 = os.path.join(d, "phase0"), os.path.join(d, "phase1")
    for sub in ("phase0/test", "phase0/pool", "phase0/data", "phase1/hard/out", "phase1/data", "self/data"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    test = fake(rng, 200, lambda i, m: 0.6, neutral=True)
    for m in list(range(4)) + ["neutral"]:
        name = "test_neutral" if m == "neutral" else f"test_strategy{m}"
        write_gz(f"{p0}/test/{name}.jsonl.gz", [{"id": i, "mode": m, "texts": [""] * len(v[m]),
                                                 "answers": [a for a, _ in v[m]], "correct": [c for _, c in v[m]]}
                                                for i, v in test.items()])
    pool = fake(rng, 300, lambda i, m: 0.3 + 0.1 * m)
    pool = {"p" + k: v for k, v in pool.items()}
    for shard, ids in enumerate((sorted(pool)[:150], sorted(pool)[150:])):
        d0 = f"{p0}/pool" if shard == 0 else f"{p1}/hard/out"
        write_gz(f"{d0}/pool_{shard}.jsonl.gz", [{"id": i, "mode": m, "texts": [""] * 2,
                                                  "answers": [a for a, _ in pool[i][m]],
                                                  "correct": [c for _, c in pool[i][m]]}
                                                 for m in range(4) for i in ids])
    with open(f"{p0}/data/pool.jsonl", "w") as f:
        for k, i in enumerate(sorted(pool)):
            f.write(json.dumps({"id": i, "source": ["gsm8k", "math"][k % 2]}) + "\n")
    with open(f"{p1}/data/val.jsonl", "w") as f:
        for i in sorted(pool)[:100]:
            f.write(json.dumps({"id": i}) + "\n")
    with open(f"{d}/self/data/hv.jsonl", "w") as f:
        for st in ["A"] * 3 + ["B0"] * 2:
            f.write(json.dumps({"id": st, "stratum": st}) + "\n")
    env = dict(os.environ, PYTHONUTF8="1")
    subprocess.run([sys.executable, "-m", "phase1.strat_overlap", "--phase0", p0, "--phase1", p1,
                    "--hv", f"{d}/self/data/hv.jsonl"], check=True, env=env)
    md = open(f"{p0}/strat_overlap.md", encoding="utf-8").read()
    js = json.load(open(f"{p0}/strat_overlap.json"))
    assert set(js["sets"]) == {"test", "val", "pool"}, js["sets"].keys()
    assert js["sets"]["test"]["all"]["problems"] == 200 and js["sets"]["pool"]["all"]["problems"] == 300
    assert js["sets"]["val"]["all"]["problems"] == 100 and set(js["sets"]["pool"]) == {"all", "gsm8k", "math"}
    assert js["hv"] == {"A": 3, "B0": 2}
    assert "## 2." in md and "## 6." in md and "| test |" in md


def main():
    test_estimators()
    test_gain()
    d = tempfile.mkdtemp()
    try:
        test_script(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("all ok")


if __name__ == "__main__":
    main()
