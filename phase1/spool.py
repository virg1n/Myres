"""Problems for multi-round self-training (phase1/srun.py): none of them had a correct solution in phase-1 training.

  cands   A: problems of the phase-0 pool and of the phase-1 hard pool with 0 correct base solutions over all their
             attempts (8 or 16; never in train.jsonl), one per question, none whose question phase-1 training,
             selection or a test used under another id; "Av" if the question was in a verifier example of phase-1
             training (a wrong solution labelled "No"): those go to the pool only, never to the held-out set;
          B: fresh problems: MATH train level 4-5 left over by phases 0-1 (then MATH test level 5 if --math_test;
             the phase-0 test uses MATH test levels 3-4), synthetic train-template problems with --ops operations
             (a new seed); exact duplicates of any earlier question are dropped.
          B is written in --shards shards for screening by the base model (phase0.gen k_solve, neutral prompt).
  select  B by its base successes of 8: 0 -> B0, 1-2 -> B12, 3-6 -> B36 (medium), 7-8 dropped.
          hard = A + B0 + B12, split by (stratum, source) into hv (held out, never trained on) and the pool;
          the pool adds the medium problems (support for the bank: otherwise a few hundred problems);
          frozen = pool-hard problems on which every start model gets extra attempts without training.
          Writes <out>/data/{pool,hv,frozen,test}.jsonl and <out>/data/meta.json."""
import argparse
import gzip
import json
import os
import random
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase0 import data as D  # noqa: E402
from phase0 import synth  # noqa: E402

KEYS = ("id", "source", "bin", "question", "answer")


def read_gz(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(l) for l in f]


def load_jsonl(path):
    return D.load_jsonl(path) if os.path.exists(path) else []


def norm_q(q):
    return re.sub(r"\s+", " ", q).strip().lower()


def base_counts(dirs):
    """id -> [correct, attempts] over every pool_*.jsonl.gz of the given directories."""
    k = {}
    for d in dirs:
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if f.startswith("pool_") and f.endswith(".jsonl.gz"):
                for r in read_gz(os.path.join(d, f)):
                    c = k.setdefault(r["id"], [0, 0])
                    c[0] += sum(r["correct"])
                    c[1] += len(r["correct"])
    return k


def known(a):
    """Every problem of phases 0 and 1 (id -> problem); the questions that phase-1 training, selection or the tests
    used (normalized); the questions of the verifier examples of phase-1 training."""
    probs = {}
    for f in (f"{a.phase0}/data/pool.jsonl", f"{a.phase1}/hard/pool.jsonl", f"{a.phase0}/data/test.jsonl",
              f"{a.phase0}/data/val.jsonl", f"{a.phase1}/data/test.jsonl", f"{a.phase1}/data/val.jsonl"):
        for p in load_jsonl(f):
            probs.setdefault(p["id"], p)
    used = {norm_q(p["question"]) for f in (f"{a.phase1}/data/train.jsonl", f"{a.phase1}/data/val.jsonl",
                                            f"{a.phase1}/data/test.jsonl", f"{a.phase0}/data/test.jsonl",
                                            f"{a.phase0}/data/val.jsonl") for p in load_jsonl(f)}
    verified = {norm_q(v["question"]) for v in load_jsonl(f"{a.phase1}/data/verify.jsonl")}
    return probs, used, verified


def cmd_cands(a):
    out = os.path.join(a.out, "pool")
    os.makedirs(out, exist_ok=True)
    probs, used, verified = known(a)
    pool_ids = {p["id"] for f in (f"{a.phase0}/data/pool.jsonl", f"{a.phase1}/hard/pool.jsonl") for p in load_jsonl(f)}
    cnt = base_counts([f"{a.phase0}/pool", f"{a.phase1}/hard/out"])
    # A: by question, not by id: a question that training, selection or a test used under another id is out, and
    # a question that was a verifier example ("Av") goes to the pool only, never to the held-out set
    A, qa = [], set()
    for i in sorted(pool_ids):
        q = norm_q(probs[i]["question"])
        if i in cnt and cnt[i][0] == 0 and cnt[i][1] >= 8 and q not in used and q not in qa:
            qa.add(q)
            A.append(dict({k: probs[i][k] for k in KEYS}, stratum="Av" if q in verified else "A", screen=cnt[i]))
    seen = {norm_q(p["question"]) for p in probs.values()} | used
    B, notes = [], []
    if a.n_math:
        try:
            m = [p for p in D.load_math("train", (4, 5), notes) if p["id"] not in probs]
            if a.math_test:
                m += D.load_math("test", (5,), notes)
        except Exception as e:           # no dataset cache and no network: synthetic problems only
            notes.append(f"MATH не загрузился ({type(e).__name__}: {e}); только синтетика")
            m = []
        B += m[:a.n_math] if a.n_math > 0 else m
    lo, hi = a.ops
    for i, r in enumerate(synth.make_set(a.seed, "train", a.n_synth, lo, hi)):
        B.append({"id": f"synth-self-{i}", "source": "synth", "bin": f"synth_{r['ops']}", "question": r["question"],
                  "answer": r["answer"]})
    fresh = []
    for p in B:
        q = norm_q(p["question"])
        if p["id"] not in probs and q not in seen:
            seen.add(q)
            fresh.append({k: p[k] for k in KEYS})
    D.save_jsonl(os.path.join(out, "A.jsonl"), A)
    test = load_jsonl(f"{a.phase1}/data/test.jsonl")
    for s in range(a.shards):
        sd = os.path.join(out, "cand", f"shard{s}")
        os.makedirs(sd, exist_ok=True)
        D.save_jsonl(os.path.join(sd, "test.jsonl"), fresh[s::a.shards])     # k_solve solves <data>/test.jsonl
    json.dump({"A": len(A), "A_verifier": sum(p["stratum"] == "Av" for p in A), "B": len(fresh), "B_by_source": {s: sum(p["source"] == s for p in fresh)
                                                             for s in sorted({p["source"] for p in fresh})},
               "notes": notes, "test_problems": len(test)}, open(os.path.join(out, "cands.json"), "w"), indent=1)
    print(f"candidates: A {len(A)} (0 correct base solutions), B {len(fresh)} fresh {notes}", flush=True)
    open(os.path.join(out, "DONE"), "w").close()


def cmd_screen(a):
    """The base model solves shard --shard of B 8 times (neutral prompt)."""
    from phase1.evaluate import gen
    out = os.path.join(a.out, "pool", "screen", f"shard{a.shard}")
    gen("k_solve", a.model, os.path.join(a.out, "pool", "cand", f"shard{a.shard}"), out,
        ["--tag", "0", "--n", "8", "--seed", str(a.seed + a.shard)], max_tokens=a.max_tokens)
    open(os.path.join(out, "DONE"), "w").close()


def stratum(ok, n):
    if n == 0:
        return None
    r = ok / n
    return "B0" if ok == 0 else "B12" if r <= 0.25 else "B36" if r <= 0.75 else None


def split(rows, n_take, rng):
    """Random split proportional by (stratum, source), exactly n_take rows taken (largest remainders)."""
    groups = {}
    for p in rows:
        groups.setdefault((p["stratum"], p["source"]), []).append(p)
    keys = sorted(groups)
    quota = {g: n_take * len(groups[g]) / max(1, len(rows)) for g in keys}
    k = {g: int(quota[g]) for g in keys}
    for g in sorted(keys, key=lambda g: (-(quota[g] - k[g]), g))[:n_take - sum(k.values())]:
        k[g] += 1
    take, rest = [], []
    for g in keys:
        lst = groups[g]
        rng.shuffle(lst)
        take += lst[:k[g]]
        rest += lst[k[g]:]
    return take, rest


def cmd_select(a):
    out = os.path.join(a.out, "data")
    os.makedirs(out, exist_ok=True)
    pd = os.path.join(a.out, "pool")
    rng = random.Random(a.seed)
    A = D.load_jsonl(os.path.join(pd, "A.jsonl"))
    B = []
    for s in range(a.shards):
        cand = {p["id"]: p for p in D.load_jsonl(os.path.join(pd, "cand", f"shard{s}", "test.jsonl"))}
        f = os.path.join(pd, "screen", f"shard{s}", "solve_0_test.jsonl.gz")
        rows = read_gz(f) if cand else []
        if sorted(r["id"] for r in rows) != sorted(cand) or any(len(r["correct"]) != 8 for r in rows):
            raise SystemExit(f"{f}: the screening must have 8 attempts of every candidate of shard {s}")
        for r in rows:
            st = stratum(sum(r["correct"]), len(r["correct"]))
            if st:
                B.append(dict(cand[r["id"]], stratum=st, screen=[sum(r["correct"]), len(r["correct"])]))
    hard = [p for p in A if p["stratum"] == "A"] + [p for p in B if p["stratum"] in ("B0", "B12")]
    med = [p for p in B if p["stratum"] == "B36"]
    hv, pool_hard = split(hard, min(a.n_hv, len(hard) // 3), rng)
    pool_hard += [p for p in A if p["stratum"] == "Av"]         # seen as verifier questions: never held out
    rng.shuffle(pool_hard)
    pool_hard = pool_hard[:a.n_hard]
    rng.shuffle(med)
    med = med[:a.n_med]
    pool = sorted(pool_hard + med, key=lambda p: p["id"])
    frozen = sorted(rng.sample(pool_hard, min(a.n_frozen, len(pool_hard))), key=lambda p: p["id"])
    hv = sorted(hv, key=lambda p: p["id"])
    for name, rows in (("pool", pool), ("hv", hv), ("frozen", frozen)):
        D.save_jsonl(os.path.join(out, f"{name}.jsonl"), rows)
    shutil.copy(os.path.join(a.phase1, "data", "test.jsonl"), os.path.join(out, "test.jsonl"))

    qs = {name: {norm_q(p["question"]) for p in rows} for name, rows in (("pool", pool), ("hv", hv))}
    assert not qs["pool"] & qs["hv"], "a question in both the pool and the held-out set"

    def by(rows):
        return {st: sum(p["stratum"] == st for p in rows) for st in ("A", "Av", "B0", "B12", "B36")}
    meta = {"pool": by(pool), "hv": by(hv), "frozen": by(frozen), "sizes": {"pool": len(pool), "hv": len(hv),
            "frozen": len(frozen), "test": sum(1 for _ in open(os.path.join(out, "test.jsonl")))},
            "sources": {name: {s: sum(p["source"] == s for p in rows) for s in sorted({p["source"] for p in rows})}
                        for name, rows in (("pool", pool), ("hv", hv))}}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print("self-training data:", json.dumps(meta), flush=True)
    if not pool or not hv:
        raise SystemExit("the pool or the held-out set is empty: not enough hard problems")
    open(os.path.join(out, "DONE"), "w").close()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("cands", "screen", "select"):
        p = sub.add_parser(name)
        p.add_argument("--phase0", default="")
        p.add_argument("--phase1", default="")
        p.add_argument("--out", required=True)
        p.add_argument("--shards", type=int, default=4)
        p.add_argument("--seed", type=int, default=1001)
    g = sub.choices["screen"]
    g.add_argument("--model", required=True)
    g.add_argument("--shard", type=int, required=True)
    g.add_argument("--max_tokens", type=int, default=1200)
    c = sub.choices["cands"]
    c.add_argument("--n_math", type=int, default=-1, help="fresh MATH problems (-1: all, 0: none)")
    c.add_argument("--math_test", type=int, default=1, help="1: also MATH test level 5 (not in the phase-0 test)")
    c.add_argument("--n_synth", type=int, default=5000)
    c.add_argument("--ops", type=int, nargs=2, default=(8, 13))
    s = sub.choices["select"]
    s.add_argument("--n_hv", type=int, default=800)
    s.add_argument("--n_hard", type=int, default=1800)
    s.add_argument("--n_med", type=int, default=600)
    s.add_argument("--n_frozen", type=int, default=600)
    a = ap.parse_args()
    if a.cmd != "screen" and not (a.phase0 and a.phase1):
        raise SystemExit(f"{a.cmd} needs --phase0 and --phase1")
    {"cands": cmd_cands, "screen": cmd_screen, "select": cmd_select}[a.cmd](a)


if __name__ == "__main__":
    main()
