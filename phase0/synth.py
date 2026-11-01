"""Synthetic multi-step word problems with a train/test split by templates.

Each problem: people hold some items; one count is given, the others are defined from earlier
ones ("X has 3 more than Y", "twice as many as", "as many as Y and Z combined", ...). The question
asks for one count. Difficulty = number of arithmetic operations needed.
Train and test use disjoint phrasings, names and item types, so test problems are new templates."""
import random

NAMES = {
    "train": ["Anna", "Ben", "Carla", "David", "Elena", "Felix", "Grace", "Hugo", "Iris", "Jonas",
              "Kira", "Leo", "Maya", "Nico", "Olga", "Pavel", "Rosa", "Sam", "Tara", "Victor"],
    "test": ["Amir", "Bella", "Chen", "Dina", "Emil", "Fatima", "Gleb", "Hana", "Ivan", "Julia",
             "Kofi", "Lena", "Marek", "Nadia", "Omar", "Priya", "Ruslan", "Sofia", "Tomas", "Yuki"],
}
ITEMS = {
    "train": ["marbles", "stamps", "stickers", "coins", "books", "pencils", "shells", "cards"],
    "test": ["postcards", "buttons", "seeds", "badges", "keys", "ribbons", "pebbles", "tokens"],
}
# relation -> phrasings per split; {x},{y},{z} names, {n} number, {it} item
PHRASES = {
    "given": {"train": ["{x} has {n} {it}."],
              "test": ["{x} owns exactly {n} {it}."]},
    "more": {"train": ["{x} has {n} more {it} than {y}."],
             "test": ["{x} owns {n} {it} more than {y} does."]},
    "fewer": {"train": ["{x} has {n} fewer {it} than {y}."],
              "test": ["{x} owns {n} {it} less than {y} does."]},
    "times": {"train": ["{x} has {n} times as many {it} as {y}."],
              "test": ["The number of {it} {x} owns is {n} times the number {y} owns."]},
    "half": {"train": ["{x} has half as many {it} as {y}."],
             "test": ["{x} owns exactly half of what {y} owns in {it}."]},
    "sum": {"train": ["{x} has as many {it} as {y} and {z} combined."],
            "test": ["{x} owns as many {it} as {y} and {z} own together."]},
    "diff": {"train": ["{x} has as many {it} as {y} has more than {z}."],
             "test": ["The number of {it} {x} owns equals how many more {y} owns than {z}."]},
}
QUESTION = {"train": "How many {it} does {x} have?", "test": "How many {it} does {x} own?"}


def _define(rng, kind, vals, known, chain=True):
    """Pick operands for a relation; returns (value, args) or None if invalid.

    chain=True uses the most recent person as the main operand, so every person is needed."""
    y = known[-1] if chain else rng.choice(known)
    if kind == "more":
        n = rng.randint(2, 30)
        return vals[y] + n, (y, None, n)
    if kind == "fewer":
        if vals[y] < 3:
            return None
        n = rng.randint(1, vals[y] - 1)
        return vals[y] - n, (y, None, n)
    if kind == "times":
        n = rng.randint(2, 5)
        if vals[y] * n > 2000:
            return None
        return vals[y] * n, (y, None, n)
    if kind == "half":
        if vals[y] % 2 or vals[y] < 2:
            return None
        return vals[y] // 2, (y, None, None)
    if len(known) < 2:
        return None
    z = rng.choice([k for k in known if k != y])
    if kind == "sum":
        return vals[y] + vals[z], (y, z, None)
    if vals[y] <= vals[z]:
        y, z = z, y
    if vals[y] == vals[z]:
        return None
    return vals[y] - vals[z], (y, z, None)


def generate(rng, ops, split):
    """One problem needing exactly `ops` operations (plus 1-2 distractor sentences)."""
    while True:
        names = rng.sample(NAMES[split], min(ops + 3, len(NAMES[split])))
        it = rng.choice(ITEMS[split])
        vals, deps, sents, order = {}, {}, [], []
        base = names[0]
        vals[base] = rng.randint(4, 40)
        deps[base] = []
        sents.append(("given", base, None, None, vals[base]))
        order.append(base)
        ok = True
        for x in names[1:ops + 1]:
            for _ in range(20):
                kind = rng.choice(["more", "fewer", "times", "half", "sum", "diff"])
                r = _define(rng, kind, vals, order)
                if r is not None:
                    break
            else:
                ok = False
                break
            v, (y, z, n) = r
            vals[x] = v
            deps[x] = [d for d in (y, z) if d]
            sents.append((kind, x, y, z, n))
            order.append(x)
        if not ok:
            continue
        target = order[-1]
        need = _closure(deps, target)
        n_ops = len(need) - 1
        if n_ops != ops:
            continue
        # distractors: defined from needed people but not used by the target
        for x in names[ops + 1:ops + 1 + rng.randint(1, 2)]:
            kind = rng.choice(["more", "fewer", "times"])
            r = _define(rng, kind, vals, order, chain=False)
            if r is None:
                continue
            v, (y, z, n) = r
            vals[x] = v
            sents.append((kind, x, y, z, n))
        text = _render(rng, sents, it, split)
        q = QUESTION[split].format(it=it, x=target)
        return {"question": text + " " + q, "answer": str(vals[target]), "ops": ops}


def _closure(deps, x):
    seen, st = set(), [x]
    while st:
        q = st.pop()
        if q not in seen:
            seen.add(q)
            st.extend(deps.get(q, []))
    return seen


def _render(rng, sents, it, split):
    out = []
    for kind, x, y, z, n in sents:
        tpl = rng.choice(PHRASES[kind][split])
        out.append(tpl.format(x=x, y=y, z=z, n=n, it=it))
    first, rest = out[0], out[1:]
    rng.shuffle(rest)
    return " ".join([first] + rest)


def make_set(seed, split, n, op_lo, op_hi):
    rng = random.Random(seed)
    ops = list(range(op_lo, op_hi + 1))
    return [dict(generate(rng, ops[i % len(ops)], split), id=f"synth-{split}-{i}") for i in range(n)]
