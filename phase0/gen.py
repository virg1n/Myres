"""GPU jobs of phase 0. Each process loads exactly one model.

  base_test    base model on the test set: neutral prompt (8 samples) + 4 strategies (2 each),
               then verification: neutral verifier on the neutral solutions, strategy verifier j on
               strategy solutions i (full 4x4 matrix)
  noise_val    greedy accuracy on val (for choosing the noise scale)
  noise_solve  noisy model on the test set, neutral prompt, 2 samples
  noise_verify noisy model j verifies sample 0 of every noisy model i (neutral verifier)
  pool         base model on a shard of the train pool: 4 strategies x 2 samples
  mode_solve, mode_verify, single_eval, k_solve   phase-1 evaluations (phase1/evaluate.py, phase1/keval.py)
"""
import argparse
import gzip
import json
import os
import time

from . import answers, prompts
from .backend import load_backend
from .data import load_jsonl

TEMP, TOP_P, TOP_K = 0.7, 0.8, 20       # Qwen3 recommended sampling for non-thinking mode
MAX_TOKENS = 1200


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def write_gz(path, rows):
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_gz(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(l) for l in f]


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def grade(probs, texts_per_problem, mode):
    rows = []
    for p, texts in zip(probs, texts_per_problem):
        keys = [answers.key(answers.extract(t)) for t in texts]
        truth = answers.key(p["answer"])
        rows.append({"id": p["id"], "mode": mode, "texts": texts, "answers": keys,
                     "correct": [k == truth and k != "" for k in keys]})
    return rows


def solve(be, tok, probs, mode, n, temperature=TEMP, prefix=None):
    ps = [prompts.solve_prompt(tok, p["question"], mode, prefix) for p in probs]
    t0 = time.time()
    texts = be.generate(ps, n, temperature, TOP_P, TOP_K, MAX_TOKENS)
    rows = grade(probs, texts, mode)
    acc = sum(sum(r["correct"]) for r in rows) / max(1, sum(len(r["correct"]) for r in rows))
    log(f"solve mode={mode} n={n}: {len(probs)} problems, acc {acc:.3f}, {time.time() - t0:.0f}s")
    return rows


def verify(be, tok, probs, sols, mode, prefix=None):
    """P(No) of verifier `mode` for every solution text in sols (list of str per problem)."""
    ps = [prompts.verify_prompt(tok, p["question"], s, mode, prefix) for p, s in zip(probs, sols)]
    t0 = time.time()
    res, missing = be.p_no(ps)
    log(f"verify mode={mode}: {len(ps)} solutions, {time.time() - t0:.0f}s, yes/no missing {missing}")
    return res, missing


def main():
    global MAX_TOKENS
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=["base_test", "noise_val", "noise_solve", "noise_verify", "pool",
                                    "mode_solve", "mode_verify", "single_eval", "k_solve"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--eager", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--noise_dir", default="", help="noise_verify: directory with solve_<i>.jsonl.gz")
    ap.add_argument("--max_tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--tag", default="", help="noise jobs: member / sigma tag used in file names")
    ap.add_argument("--members", type=int, default=4, help="noise_verify / mode_verify: number of members")
    ap.add_argument("--n", type=int, default=2, help="pool / mode_solve: samples per prompt")
    ap.add_argument("--sets", default="test", help="mode_solve / single_eval: comma-separated set names")
    ap.add_argument("--solve_dir", default="", help="mode_verify: directory with solve_<i>_test.jsonl.gz")
    ap.add_argument("--prefix_mode", type=int, default=-1, help="mode_solve / mode_verify: ModC prefix index")
    ap.add_argument("--greedy", type=int, default=0, help="k_solve: also one greedy (T = 0) solution per problem")
    a = ap.parse_args()
    MAX_TOKENS = a.max_tokens
    prefix = a.prefix_mode if a.prefix_mode >= 0 else None
    os.makedirs(a.out, exist_ok=True)
    be = load_backend(a.backend, a.model, seed=a.seed, eager=bool(a.eager))
    tok = be.tok
    test = load_jsonl(os.path.join(a.data, "test.jsonl"))
    t0 = time.time()

    if a.job == "base_test":
        f = os.path.join(a.out, "test_neutral.jsonl.gz")
        if not os.path.exists(f):
            write_gz(f, solve(be, tok, test, "neutral", 8))
        for k in range(4):
            f = os.path.join(a.out, f"test_strategy{k}.jsonl.gz")
            if not os.path.exists(f):
                write_gz(f, solve(be, tok, test, k, 2))
        f = os.path.join(a.out, "verify_neutral.json")
        if not os.path.exists(f):
            neu = read_gz(os.path.join(a.out, "test_neutral.jsonl.gz"))
            mat, miss = [], 0
            for i in range(4):                       # "modes" of the baseline = samples 0..3
                r, m = verify(be, tok, test, [x["texts"][i] for x in neu], "neutral")
                mat.append(r)
                miss += m
            write_json(f, {"p_no": mat, "missing": miss})       # [i][n]
        f = os.path.join(a.out, "verify_strategies.json")
        if not os.path.exists(f):
            st = [read_gz(os.path.join(a.out, f"test_strategy{k}.jsonl.gz")) for k in range(4)]
            mat, miss = [[None] * 4 for _ in range(4)], 0
            for i in range(4):
                sols = [x["texts"][0] for x in st[i]]
                for j in range(4):
                    mat[i][j], m = verify(be, tok, test, sols, j)
                    miss += m
            write_json(f, {"p_no": mat, "missing": miss})       # [i][j][n]

    elif a.job == "noise_val":
        val = load_jsonl(os.path.join(a.data, "val.jsonl"))
        rows = solve(be, tok, val, "neutral", 1, temperature=0.0)
        acc = sum(r["correct"][0] for r in rows) / len(rows)
        write_json(os.path.join(a.out, f"val_{a.tag}.json"), {"acc": acc})

    elif a.job == "noise_solve":
        write_gz(os.path.join(a.out, f"solve_{a.tag}.jsonl.gz"), solve(be, tok, test, "neutral", 2))

    elif a.job == "noise_verify":
        res, miss = [], 0
        for i in range(a.members):
            sols = [x["texts"][0] for x in read_gz(os.path.join(a.noise_dir, f"solve_{i}.jsonl.gz"))]
            r, m = verify(be, tok, test, sols, "neutral")
            res.append(r)
            miss += m
        write_json(os.path.join(a.out, f"verify_{a.tag}.json"), {"p_no": res, "missing": miss})   # [i][n]

    elif a.job == "pool":
        pool = load_jsonl(os.path.join(a.data, "pool.jsonl"))
        part = pool[a.shard::a.nshards]
        rows = []
        for k in range(4):
            rows += solve(be, tok, part, k, a.n)
        write_gz(os.path.join(a.out, f"pool_{a.shard}.jsonl.gz"), rows)

    elif a.job == "mode_solve":                     # one mode of a phase-1 model, neutral prompt
        for s in a.sets.split(","):
            f = os.path.join(a.out, f"solve_{a.tag}_{s}.jsonl.gz")
            if not os.path.exists(f):
                write_gz(f, solve(be, tok, load_jsonl(os.path.join(a.data, f"{s}.jsonl")), "neutral", a.n, prefix=prefix))

    elif a.job == "mode_verify":                    # mode `tag` verifies sample 0 of every mode i on test
        res, miss = [], 0
        for i in range(a.members):
            sols = [x["texts"][0] for x in read_gz(os.path.join(a.solve_dir, f"solve_{i}_test.jsonl.gz"))]
            r, m = verify(be, tok, test, sols, "neutral", prefix)
            res.append(r)
            miss += m
        write_json(os.path.join(a.out, f"verify_{a.tag}.json"), {"p_no": res, "missing": miss})   # [i][n]

    elif a.job == "k_solve":                        # one unit of phase1.keval: n samples on test (+ greedy)
        f = os.path.join(a.out, f"solve_{a.tag}_test.jsonl.gz")
        if not os.path.exists(f):
            write_gz(f, solve(be, tok, test, "neutral", a.n))
        f = os.path.join(a.out, f"greedy_{a.tag}_test.jsonl.gz")
        if a.greedy and not os.path.exists(f):
            write_gz(f, solve(be, tok, test, "neutral", 1, temperature=0.0))

    elif a.job == "single_eval":                    # model without modes: 8 samples, one verifier
        for s in a.sets.split(","):
            f = os.path.join(a.out, f"solve_single_{s}.jsonl.gz")
            if not os.path.exists(f):
                write_gz(f, solve(be, tok, load_jsonl(os.path.join(a.data, f"{s}.jsonl")), "neutral", 8))
        neu = read_gz(os.path.join(a.out, "solve_single_test.jsonl.gz"))
        mat, miss = [], 0
        for i in range(4):
            r, m = verify(be, tok, test, [x["texts"][i] for x in neu], "neutral")
            mat.append(r)
            miss += m
        write_json(os.path.join(a.out, "verify_single.json"), {"p_no": mat, "missing": miss})

    log(f"{a.job} finished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
