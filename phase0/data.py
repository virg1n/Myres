"""Test / val / train-pool sets for phase 0.

test : GSM-Symbolic P1+P2 (new variants of GSM8K-test templates), MATH test levels 3-4 with numeric
       answers, synthetic problems on held-out templates (ops 4-8).
val  : small disjoint set for tuning the noise scale.
pool : GSM8K train, MATH train levels 1-4 (numeric), synthetic train templates (ops 2-7) --
       solved 8 times each to get multi-reference solutions and natural errors for phase 1.
Each record: id, source, bin (difficulty group), question, answer."""
import json
import random

from . import answers, synth

MATH_SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                 "number_theory", "prealgebra", "precalculus"]


def _load(name, config, split):
    from datasets import load_dataset
    return load_dataset(name, config, split=split) if config else load_dataset(name, split=split)


def load_gsm8k(split):
    ds = _load("openai/gsm8k", "main", split)
    return [{"id": f"gsm8k-{split}-{i}", "source": "gsm8k", "bin": "gsm8k", "question": r["question"],
             "answer": answers.gsm_truth(r["answer"])} for i, r in enumerate(ds)]


def load_gsm_symbolic(variant, notes):
    try:
        ds = _load("apple/GSM-Symbolic", variant, "test")
    except Exception as e:
        notes.append(f"GSM-Symbolic {variant} не загрузился ({type(e).__name__}); вместо него GSM8K test")
        return None
    out = []
    for i, r in enumerate(ds):
        out.append({"id": f"gsmsym-{variant}-{i}", "source": "gsm_symbolic", "bin": f"gsm_{variant}",
                    "question": r["question"], "answer": answers.gsm_truth(r["answer"])})
    return out


def load_math(split, levels, notes):
    rows = []
    try:
        for subj in MATH_SUBJECTS:
            for r in _load("EleutherAI/hendrycks_math", subj, split):
                rows.append(r)
    except Exception as e:
        notes.append(f"EleutherAI/hendrycks_math не загрузился ({type(e).__name__}); пробую MATH-lighteval")
        rows = list(_load("DigitalLearningGmbH/MATH-lighteval", "default", split))
    out = []
    for i, r in enumerate(rows):
        lvl = str(r.get("level", ""))
        if not any(lvl.endswith(str(l)) for l in levels):
            continue
        truth = answers.last_boxed(r["solution"])
        if answers.to_number(truth) is None:
            continue
        out.append({"id": f"math-{split}-{i}", "source": "math", "bin": f"math_L{lvl[-1]}",
                    "question": r["problem"], "answer": truth})
    return out


def _take(rng, rows, n):
    rows = list(rows)
    rng.shuffle(rows)
    return rows[:n]


def build(n_test=1500, n_val=300, n_pool=12000, seed=0, sources=("gsm", "math", "synth")):
    """Returns dict(test, val, pool, notes). Sizes are split evenly over the sources."""
    rng = random.Random(seed)
    notes = []
    k = len(sources)
    t_each, v_each, p_each = n_test // k, n_val // k, n_pool // k
    test, val, pool = [], [], []
    if "gsm" in sources:
        sym = []
        for v in ("p1", "p2"):
            got = load_gsm_symbolic(v, notes)
            if got:
                sym.append(_take(rng, got, (t_each + v_each) // 2 + 1))
        if not sym:
            g = _take(rng, load_gsm8k("test"), t_each + v_each)
            for r in g:
                r["bin"] = "gsm8k_test"
            sym = [g]
        flat = [r for s in sym for r in s]
        rng.shuffle(flat)
        test += flat[:t_each]
        val += flat[t_each:t_each + v_each]
        pool += _take(rng, load_gsm8k("train"), p_each)
    if "math" in sources:
        mt = _take(rng, load_math("test", (3, 4), notes), t_each + v_each)
        test += mt[:t_each]
        val += mt[t_each:]
        pool += _take(rng, load_math("train", (1, 2, 3, 4), notes), p_each)
    if "synth" in sources:
        st = synth.make_set(seed + 1, "test", t_each + v_each, 4, 8)
        for r in st:
            r.update(source="synth", bin=f"synth_{r['ops']}")
        test += st[:t_each]
        val += st[t_each:]
        sp = synth.make_set(seed + 2, "train", p_each, 2, 7)
        for r in sp:
            r.update(source="synth", bin=f"synth_{r['ops']}")
        pool += sp
    for r in test + val + pool:
        r.pop("ops", None)
    return {"test": test, "val": val, "pool": pool, "notes": notes}


def save_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]
