"""Training data for phase 1 from the phase-0 pool and the extra hard pool.

For every problem: all correct self-generated solutions, grouped into structural clusters (sets of
numbers that appear, Jaccard >= 0.8 = same trajectory); one representative per cluster, at most K.
  train.jsonl   problems with >= 1 correct solution: refs (distinct, <= K), single ref
  verify.jsonl  balanced correct / natural-error solutions for the Yes/No verifier
  val.jsonl     held-out problems (not trained on) with mixed success, for choosing the stage-1 variant
  stage2.jsonl  problems where the base model succeeds 10-80% of the time
  test.jsonl    copied from phase 0 (same test -> paired comparisons)
"""
import argparse
import gzip
import json
import os
import random
import re


def read_gz(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(l) for l in f]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def save_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def numbers(text):
    return set(re.findall(r"\d+(?:\.\d+)?", text or ""))


def clusters(texts, thr=0.8):
    reps, rep_sets = [], []
    for t in texts:
        s = numbers(t)
        if not any(len(s | r) and len(s & r) / len(s | r) >= thr for r in rep_sets):
            reps.append(t)
            rep_sets.append(s)
    return reps


def collect(pool_dirs, problems):
    per = {}
    for d in pool_dirs:
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if f.startswith("pool_") and f.endswith(".jsonl.gz"):
                for r in read_gz(os.path.join(d, f)):
                    if r["id"] not in problems:
                        continue
                    e = per.setdefault(r["id"], {"ok": [], "bad": [], "n": 0})
                    e["n"] += len(r["correct"])
                    for t, a, c in zip(r["texts"], r["answers"], r["correct"]):
                        if c:
                            e["ok"].append(t)
                        elif a:
                            e["bad"].append(t)
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase0", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--n_val", type=int, default=600)
    ap.add_argument("--n_verify", type=int, default=6000)
    ap.add_argument("--n_stage2", type=int, default=3000)
    ap.add_argument("--max_train", type=int, default=0, help="0 = all")
    ap.add_argument("--max_test", type=int, default=0, help="0 = full phase-0 test (smoke runs only)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    dd = os.path.join(a.out, "data")
    os.makedirs(dd, exist_ok=True)
    test = load_jsonl(os.path.join(a.phase0, "data", "test.jsonl"))
    save_jsonl(os.path.join(dd, "test.jsonl"), test[:a.max_test] if a.max_test else test)
    problems = {p["id"]: p for p in load_jsonl(os.path.join(a.phase0, "data", "pool.jsonl"))}
    hard_f = os.path.join(a.out, "hard", "pool.jsonl")
    if os.path.exists(hard_f):
        problems.update({p["id"]: p for p in load_jsonl(hard_f)})
    per = collect([os.path.join(a.phase0, "pool"), os.path.join(a.out, "hard", "out")], problems)
    ids = sorted(per)
    rng.shuffle(ids)
    rate = {i: len(per[i]["ok"]) / max(1, per[i]["n"]) for i in ids}      # every attempt counts
    # val: held out, mixed success so that there are errors to correlate
    val_ids = [i for i in ids if 0.1 <= rate[i] <= 0.9][:a.n_val]
    vs = set(val_ids)
    train, verify, stage2 = [], [], []
    for i in ids:
        if i in vs:
            continue
        p, e = problems[i], per[i]
        if e["ok"]:
            refs = clusters(e["ok"])[:a.K]
            train.append({"id": i, "source": p["source"], "question": p["question"], "answer": p["answer"],
                          "refs": refs, "single": rng.choice(e["ok"]), "rate": rate[i]})
        if 0.1 <= rate[i] <= 0.8:
            stage2.append({k: p[k] for k in ("id", "source", "bin", "question", "answer")})
        for t in e["bad"][:2]:
            verify.append({"question": p["question"], "solution": t, "label": "No"})
        for t in e["ok"][:1]:
            verify.append({"question": p["question"], "solution": t, "label": "Yes"})
    if a.max_train:
        train = train[:a.max_train]
    no = [v for v in verify if v["label"] == "No"]
    yes = [v for v in verify if v["label"] == "Yes"]
    rng.shuffle(no)
    rng.shuffle(yes)
    h = a.n_verify // 2
    verify = no[:h] + yes[:h]
    rng.shuffle(verify)
    rng.shuffle(stage2)
    save_jsonl(os.path.join(dd, "train.jsonl"), train)
    save_jsonl(os.path.join(dd, "verify.jsonl"), verify)
    save_jsonl(os.path.join(dd, "val.jsonl"), [{k: problems[i][k] for k in ("id", "source", "bin", "question", "answer")}
                                               for i in val_ids])
    save_jsonl(os.path.join(dd, "stage2.jsonl"), stage2[:a.n_stage2])
    nref = [len(t["refs"]) for t in train]
    stats = {"train": len(train), "verify": len(verify), "val": len(val_ids), "stage2": min(len(stage2), a.n_stage2),
             "refs_hist": [nref.count(k) for k in range(1, a.K + 1)],
             "refs_mean": sum(nref) / max(1, len(nref))}
    json.dump(stats, open(os.path.join(dd, "stats.json"), "w"), indent=1)
    open(os.path.join(a.out, "data", "DONE"), "w").close()
    print("phase-1 data:", stats, flush=True)


if __name__ == "__main__":
    main()
