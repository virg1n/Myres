"""iGSM-style synthetic math problems.

A problem is a 4-layer hierarchy (towns -> buildings -> rooms -> objects).
Instance params  I(A,B): "number of B in each A" (B is a child of A).
Abstract params  T(A,C): "total number of C in each A" (C two or three layers below A),
                 implicitly = sum over children B of I(A,B) * N(B,C).
Instance params are either constants or defined from other params with + - *.
All arithmetic is mod 23.

The module also renders solutions in two styles (forward / backward), injects errors
for verification data and checks arbitrary generated solutions step by step.
"""
import hashlib
import random

MOD = 23
MAX_STEPS = 64

LAYER_NAMES = [
    ["Ashford", "Brookvale", "Cedarton", "Dunmore", "Elmwood", "Fairhaven", "Glenrock", "Hillcrest",
     "Ivydale", "Juniper", "Kingsbury", "Lakemont", "Maplewood", "Northgate", "Oakridge", "Pinecrest",
     "Quarryville", "Riverton", "Stonebridge", "Thornbury", "Upton", "Valemont", "Westbrook", "Yarrow"],
    ["school", "hospital", "library", "museum", "bakery", "factory", "garage", "theater", "hotel",
     "office", "station", "warehouse", "clinic", "gym", "market", "church", "castle", "farm", "barn",
     "studio"],
    ["classroom", "kitchen", "hallway", "closet", "cellar", "attic", "lobby", "ward", "vault", "lab",
     "pantry", "gallery", "locker", "cabinet", "shelf", "drawer", "crate", "box", "bin", "chest"],
    ["chair", "lamp", "desk", "book", "cup", "plate", "bottle", "pencil", "clock", "rug", "mirror",
     "candle", "basket", "bucket", "hammer", "brush", "fan", "pillow", "towel", "vase"],
]
OPS = ["plus", "minus", "times"]

# ----------------------------------------------------------------------------- vocabulary
SPECIAL = ["<pad>", "<bos>", "<eos>", "<solve>", "<verify>"]
WORDS = ["each", "has", "as", "many", "total", "in", "plus", "minus", "times", "=", ";", ".",
         "question", "how", "?", "sol", "ans", "need", ":", ",", "ver", "ok", "err"]
NUMS = [str(i) for i in range(MOD)]
STEP_TOKS = [f"#{i}" for i in range(1, MAX_STEPS + 1)]
ITOS = SPECIAL + WORDS + NUMS + STEP_TOKS + [n for layer in LAYER_NAMES for n in layer]
STOI = {s: i for i, s in enumerate(ITOS)}
assert len(STOI) == len(ITOS), "duplicate tokens in vocabulary"
VOCAB_SIZE = len(ITOS)
PAD, BOS, EOS, SOLVE, VERIFY = (STOI[s] for s in SPECIAL)
SOL, ANS, VER, OK, ERR, SEMI, EQ, NEED = (STOI[s] for s in ["sol", "ans", "ver", "ok", "err", ";", "=", "need"])
NUM_IDS = [STOI[s] for s in NUMS]
STEP_IDS = [STOI[s] for s in STEP_TOKS]
ID2NUM = {STOI[s]: i for i, s in enumerate(NUMS)}
ID2STEP = {STOI[s]: i + 1 for i, s in enumerate(STEP_TOKS)}
LAYER_OF = {n: li for li, layer in enumerate(LAYER_NAMES) for n in layer}


def encode(words):
    return [STOI[w] for w in words]


def decode(ids):
    return " ".join(ITOS[i] for i in ids)


# ----------------------------------------------------------------------------- problem
class Param:
    __slots__ = ("pid", "kind", "a", "b", "dtype", "const", "op", "args", "terms", "deps", "own_ops")

    def __init__(self, pid, kind, a, b):
        self.pid, self.kind, self.a, self.b = pid, kind, a, b
        self.dtype = None      # inst: "const" | "bin"; abs: "abs"
        self.const = None
        self.op = None
        self.args = None       # [("p", pid) | ("c", int), ...] for "bin"
        self.terms = None      # [(inst_pid, child_pid), ...] for "abs"
        self.deps = []
        self.own_ops = 0

    def desc(self):
        return [self.a, self.b] if self.kind == "inst" else [self.a, "total", self.b]


class Problem:
    def __init__(self, params, query, stmt_order, op, seed):
        self.params = params
        self.query = query
        self.stmt_order = stmt_order
        self.op = op
        self.seed = seed
        self.key = {tuple(p.desc()): p.pid for p in params}
        self.values = compute_values(params, {})
        self.necessary = closure(params, query)
        self._tokens = None

    @property
    def answer(self):
        return self.values[self.query]

    def tokens(self):
        if self._tokens is None:
            words = []
            for pid in self.stmt_order:
                words += statement(self.params, self.params[pid])
            q = self.params[self.query]
            words += ["question", "how", "many"] + (["total", q.b] if q.kind == "abs" else [q.b]) + ["in", "each", q.a, "?"]
            self._tokens = encode(words)
        return self._tokens

    def bucket(self):
        h = hashlib.blake2b(bytes(str(self.tokens()), "ascii"), digest_size=8).digest()
        return int.from_bytes(h, "little") % 1000


def statement(params, p):
    if p.dtype == "const":
        return ["each", p.a, "has", str(p.const), p.b, "."]
    x, y = p.args
    return (["each", p.a, "has", "as", "many", p.b, "as"] + operand_words(params, x) + [p.op]
            + operand_words(params, y) + ["."])


def operand_words(params, arg):
    kind, v = arg
    return [str(v)] if kind == "c" else list(params[v].desc())


def closure(params, pid):
    seen, stack = set(), [pid]
    while stack:
        q = stack.pop()
        if q in seen:
            continue
        seen.add(q)
        stack.extend(params[q].deps)
    return seen


def apply_op(op, x, y):
    if op == "plus":
        return (x + y) % MOD
    if op == "minus":
        return (x - y) % MOD
    return (x * y) % MOD


def own_value(p, vals):
    """Value of p from the (possibly wrong) values of its dependencies."""
    if p.dtype == "const":
        return p.const
    if p.dtype == "bin":
        x, y = (vals[v] if k == "p" else v for k, v in p.args)
        return apply_op(p.op, x, y)
    return sum(vals[i] * vals[c] for i, c in p.terms) % MOD


def topo_order(params, subset, rng=None):
    indeg = {q: 0 for q in subset}
    users = {q: [] for q in subset}
    for q in subset:
        for d in set(params[q].deps):
            indeg[q] += 1
            users[d].append(q)
    ready = sorted(q for q in subset if indeg[q] == 0)
    out = []
    while ready:
        i = rng.randrange(len(ready)) if rng else 0
        q = ready.pop(i)
        out.append(q)
        for u in users[q]:
            indeg[u] -= 1
            if indeg[u] == 0:
                ready.append(u)
    assert len(out) == len(subset), "cycle"
    return out


def compute_values(params, override):
    vals = {}
    for q in topo_order(params, set(range(len(params)))):
        vals[q] = override.get(q, own_value(params[q], vals))
    return vals


def _dependents(params, pid):
    """All params that (transitively) use pid, including pid itself."""
    users = {}
    for q in params:
        for d in q.deps:
            users.setdefault(d, []).append(q.pid)
    seen, stack = set(), [pid]
    while stack:
        q = stack.pop()
        if q in seen:
            continue
        seen.add(q)
        stack.extend(users.get(q, []))
    return seen


def _build(rng, op_target):
    big = op_target // 6
    sizes = [rng.randint(2, min(4, 2 + big)) for _ in range(4)]
    names = [rng.sample(LAYER_NAMES[l], sizes[l]) for l in range(4)]
    children = {}
    for l in range(3):
        for a in names[l]:
            k = rng.randint(1, min(len(names[l + 1]), 2 + big // 2))
            children[a] = rng.sample(names[l + 1], k)
    params, key = [], {}

    def add(kind, a, b):
        p = Param(len(params), kind, a, b)
        params.append(p)
        key[(kind, a, b)] = p.pid
        return p

    for l in range(3):
        for a in names[l]:
            for b in children[a]:
                add("inst", a, b)
    # reachability for abstract params, deepest first so N(B,C) exists when needed
    reach = {}
    for l in (2, 1, 0):
        for a in names[l]:
            r = set()
            for b in children[a]:
                r.add(b)
                r |= reach.get(b, set())
            reach[a] = r
    for l in (1, 0):
        for a in names[l]:
            for c in sorted(reach[a], key=lambda n: (LAYER_OF[n], n)):
                if LAYER_OF[c] < l + 2:
                    continue
                p = add("abs", a, c)
                terms = []
                for b in children[a]:
                    if c in children.get(b, []):
                        terms.append((key[("inst", a, b)], key[("inst", b, c)]))
                    elif c in reach.get(b, set()):
                        terms.append((key[("inst", a, b)], key[("abs", b, c)]))
                p.dtype, p.terms = "abs", terms
                p.deps = [x for t in terms for x in t]
                p.own_ops = 2 * len(terms) - 1
    inst = [p for p in params if p.kind == "inst"]
    rng.shuffle(inst)
    p_bin = min(0.75, 0.35 + 0.025 * op_target)
    for p in inst:
        if rng.random() > p_bin:
            p.dtype, p.const = "const", rng.randint(1, MOD - 1)
            continue
        banned = _dependents(params, p.pid)
        cands = [q.pid for q in params if q.pid not in banned]
        if not cands:
            p.dtype, p.const = "const", rng.randint(1, MOD - 1)
            continue
        x = rng.choice(cands)
        op = rng.choice(OPS)
        if rng.random() < 0.6 and len(cands) > 1:
            y = ("p", rng.choice([c for c in cands if c != x]))
        else:
            y = ("c", rng.randint(1, MOD - 1))
        p.dtype, p.op, p.args = "bin", op, [("p", x), y]
        p.deps = [x] + ([y[1]] if y[0] == "p" else [])
        p.own_ops = 1
    return params


def generate_problem(rng, op_target, max_tries=200):
    """Problem whose query needs exactly op_target operations (closest found otherwise)."""
    best = None
    for _ in range(max_tries):
        params = _build(rng, op_target)
        ops = {}
        for p in params:
            ops[p.pid] = sum(params[q].own_ops for q in closure(params, p.pid))
        exact = [pid for pid, o in ops.items() if o == op_target]
        if exact:
            query = rng.choice(exact)
            best = (params, query, op_target)
            break
        pid = min(ops, key=lambda q: abs(ops[q] - op_target))
        if best is None or abs(ops[pid] - op_target) < abs(best[2] - op_target):
            best = (params, pid, ops[pid])
    params, query, op = best
    stmt = [p.pid for p in params if p.kind == "inst"]
    rng.shuffle(stmt)
    prob = Problem(params, query, stmt, op, None)
    prob.tokens()
    return prob


# ----------------------------------------------------------------------------- solutions
def step_words(prob, p, vals, op_override=None, args_override=None, terms_override=None, value_override=None):
    """Words for one compute step of p; returns (words, value_written)."""
    out = list(p.desc()) + ["="]
    if p.dtype == "const":
        v = p.const if value_override is None else value_override
        return out + [str(v), ";"], v
    if p.dtype == "bin":
        op = op_override or p.op
        args = args_override or p.args
        xs = [vals[v] if k == "p" else v for k, v in args]
        v = apply_op(op, xs[0], xs[1])
        if value_override is not None:
            v = value_override
        out += operand_words(prob.params, args[0]) + [op] + operand_words(prob.params, args[1])
        out += ["=", str(xs[0]), op, str(xs[1]), "=", str(v), ";"]
        return out, v
    terms = terms_override or p.terms
    for t, (i, c) in enumerate(terms):
        if t:
            out.append("plus")
        out += list(prob.params[i].desc()) + ["times"] + list(prob.params[c].desc())
    out.append("=")
    for t, (i, c) in enumerate(terms):
        if t:
            out.append("plus")
        out += [str(vals[i]), "times", str(vals[c])]
    seq = [vals[i] * vals[c] % MOD for i, c in terms]
    out += ["="] + _join_plus(seq)
    while len(seq) > 1:
        seq = [(seq[0] + seq[1]) % MOD] + seq[2:]
        out += ["="] + _join_plus(seq)
    v = seq[0] if value_override is None else value_override
    if value_override is not None:
        out[-1] = str(v)
    return out + [";"], v


def _join_plus(seq):
    out = []
    for t, x in enumerate(seq):
        if t:
            out.append("plus")
        out.append(str(x))
    return out


def plan_and_order(prob, rng, style):
    params = prob.params
    if style == "forward":
        return [], topo_order(params, prob.necessary, rng)
    plan, order, seen = [], [], set()

    def visit(q):
        if q in seen:
            return
        seen.add(q)
        deps = list(dict.fromkeys(params[q].deps))
        rng.shuffle(deps)
        if deps:
            line = ["need"] + params[q].desc() + [":"]
            for t, d in enumerate(deps):
                if t:
                    line.append(",")
                line += params[d].desc()
            plan.append(line + [";"])
        for d in deps:
            visit(d)
        order.append(q)

    visit(prob.query)
    return [w for line in plan for w in line], order


def solution_words(prob, rng, style, error=None):
    """Reference solution (error=None) or a solution with one injected error.

    Returns (words, first_error_step or None). Error propagates to later steps."""
    plan, order = plan_and_order(prob, rng, style)
    words = list(plan)
    vals = {}
    err_step = None
    inject_at = None
    if error is not None:
        inject_at = _choose_injection(prob, order, rng)
        if inject_at is None:
            return None, None
    for s, q in enumerate(order):
        p = prob.params[q]
        if s == inject_at:
            w, v = _inject(prob, p, vals, order[:s], rng)
            if w is None or v == prob.values[q]:
                return None, None
            err_step = s + 1
        else:
            w, v = step_words(prob, p, vals)
        vals[q] = v
        words += w
    words += ["ans", str(vals[prob.query])]
    return words, err_step


def _choose_injection(prob, order, rng):
    return rng.randrange(len(order)) if order else None


def _inject(prob, p, vals, done, rng):
    kinds = ["arith"]
    if p.dtype == "const":
        kinds.append("const")
    if p.dtype == "bin":
        kinds += ["op", "param"]
    if p.dtype == "abs":
        kinds += ["term"] if len(p.terms) > 1 else []
        kinds += ["child"]
    kind = rng.choice(kinds)
    true_v = own_value(p, vals)
    wrong = rng.choice([x for x in range(MOD) if x != true_v])
    if kind in ("arith", "const"):
        return step_words(prob, p, vals, value_override=wrong)
    if kind == "op":
        op = rng.choice([o for o in OPS if o != p.op])
        return step_words(prob, p, vals, op_override=op)
    if kind == "param":
        slots = [i for i, (k, _) in enumerate(p.args) if k == "p"]
        alt = [q for q in done if q not in p.deps]
        if not alt:
            return step_words(prob, p, vals, value_override=wrong)
        args = list(p.args)
        args[rng.choice(slots)] = ("p", rng.choice(alt))
        return step_words(prob, p, vals, args_override=args)
    if kind == "term":
        terms = list(p.terms)
        terms.pop(rng.randrange(len(terms)))
        return step_words(prob, p, vals, terms_override=terms)
    # child: use a wrong (already computed) param in place of one term's child count
    alt = [q for q in done if q not in p.deps]
    if not alt:
        return step_words(prob, p, vals, value_override=wrong)
    terms = list(p.terms)
    t = rng.randrange(len(terms))
    terms[t] = (terms[t][0], rng.choice(alt))
    return step_words(prob, p, vals, terms_override=terms)


# ----------------------------------------------------------------------------- sequences
def solve_prompt(prob):
    return [BOS, SOLVE] + prob.tokens() + [SOL]


def solve_sequence(prob, rng, style=None):
    """Full training sequence for solving; loss starts after SOL."""
    style = style or rng.choice(["forward", "backward"])
    words, _ = solution_words(prob, rng, style)
    prompt = solve_prompt(prob)
    return prompt + encode(words) + [EOS], len(prompt)


def verdict_tokens(err_step):
    if err_step is None:
        return [VER, OK, EOS]
    return [VER, ERR, STEP_IDS[min(err_step, MAX_STEPS) - 1], EOS]


def verify_prefix(prob, sol_ids):
    """[bos verify problem sol <solution ids>] -- solution ids exclude EOS."""
    return [BOS, VERIFY] + prob.tokens() + [SOL] + list(sol_ids)


def verify_sequence(prob, rng, with_error):
    """Verification training sequence; loss only on the verdict (after VER)."""
    style = rng.choice(["forward", "backward"])
    for _ in range(10):
        words, err = solution_words(prob, rng, style, error=True if with_error else None)
        if words is not None:
            break
    else:
        words, err = solution_words(prob, rng, style)
    prefix = verify_prefix(prob, encode(words))
    return prefix + verdict_tokens(err), len(prefix) + 1


# ----------------------------------------------------------------------------- checking
def check_solution(prob, ids):
    """Check generated solution ids (after SOL, EOS optional).

    Returns dict: answer, correct (final answer), has_error, first_err (1-based step,
    n_steps+1 if only the answer is wrong/missing), err_pid, style, values, order, n_steps."""
    if ids and ids[-1] == EOS:
        ids = ids[:-1]
    ans = None
    if ANS in ids:
        k = ids.index(ANS)
        if k + 1 < len(ids) and ids[k + 1] in ID2NUM:
            ans = ID2NUM[ids[k + 1]]
        ids = ids[:k]
    segs, cur = [], []
    for t in ids:
        cur.append(t)
        if t == SEMI:
            segs.append(cur)
            cur = []
    if cur:
        segs.append(cur)
    defined, values, order, steps = set(), {}, [], []
    first_err, err_pid, style, step = None, None, "forward", 0
    for seg in segs:
        if seg[0] == NEED:
            style = "backward"
            continue
        step += 1
        pid, v = _parse_step(prob, seg)
        ok = pid is not None and v is not None and seg[-1] == SEMI
        if ok:
            ok = all(d in defined for d in prob.params[pid].deps) and v == prob.values[pid]
        if pid is not None and v is not None:
            steps.append((step, pid, v))
            values.setdefault(pid, v)
            order.append(pid)
            defined.add(pid)
        if not ok and first_err is None:
            first_err, err_pid = step, pid
    correct = ans is not None and ans == prob.answer
    if first_err is None and not correct:
        first_err = step + 1
    return {"answer": ans, "correct": correct, "has_error": first_err is not None,
            "first_err": first_err, "err_pid": err_pid, "style": style, "values": values,
            "order": order, "steps": steps, "n_steps": step}


def _parse_step(prob, seg):
    if EQ not in seg:
        return None, None
    k = seg.index(EQ)
    desc = tuple(ITOS[t] for t in seg[:k])
    pid = prob.key.get(desc)
    body = seg[:-1] if seg[-1] == SEMI else seg
    v = ID2NUM.get(body[-1]) if len(body) > k + 1 else None
    return pid, v


# ----------------------------------------------------------------------------- splits
# buckets of hash(problem text) % 1000; train uses >= 20 so eval problems are never trained on
SPLITS = {"test": (0, 5), "val": (5, 10), "natural": (10, 20), "train": (20, 1000)}


def in_split(prob, split):
    lo, hi = SPLITS[split]
    return lo <= prob.bucket() < hi


def sample_problem(rng, split, op_lo, op_hi, max_len=1024):
    """Random problem from a split with op uniform in [op_lo, op_hi] that fits max_len."""
    while True:
        op = rng.randint(op_lo, op_hi)
        prob = generate_problem(rng, op)
        if not in_split(prob, split):
            continue
        if max_seq_len(prob) > max_len:
            continue
        return prob


def max_seq_len(prob):
    """Length of the longest training sequence for prob (backward-style verification)."""
    words, _ = solution_words(prob, random.Random(0), "backward")
    return len(prob.tokens()) + len(words) + 8


def fixed_set(seed, split, n, op_lo, op_hi, max_len=1024):
    """Deterministic evaluation set; ops cycle evenly through [op_lo, op_hi]."""
    rng = random.Random(seed)
    out = []
    ops = list(range(op_lo, op_hi + 1))
    while len(out) < n:
        op = ops[len(out) % len(ops)]
        prob = sample_problem(rng, split, op, op, max_len)
        out.append(prob)
    return out
