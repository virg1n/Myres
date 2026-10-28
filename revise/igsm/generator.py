"""iGSM-style synthetic grade-school math generator (re-implementation).

Follows the construction of Ye et al., "Physics of Language Models: Part 2.1"
(2024): a hierarchical *structure graph* (items in 4 layers, edges between
consecutive layers), *instance parameters* (one per edge: "the number of each
Central High 's Film Studio"), *abstract parameters* (a whole category below an
item: "Central High 's Backpack" = sum over paths of products), and a random
*dependency graph* in which every instance parameter is a constant or a
function of other parameters. All arithmetic is mod 23. The solution computes
exactly the necessary parameters in a random topological order and introduces
temporary variables for multi-operation formulas, e.g.

  Define Film Studio 's School Daypack as g ; R = W + B = 13 + 7 = 20 ;
  so g = R + 12 = 20 + 12 = 9 .

Every quantity's true value and the exact dependency graph are kept in the
returned Problem object, which the checker (checker.py) uses for labelling.
"""

import random
from dataclasses import dataclass, field

from .tokenizer import Tokenizer
from .vocab import LETTERS, MOD, WORLDS

OPS = {"+": lambda a, b: (a + b) % MOD, "-": lambda a, b: (a - b) % MOD, "*": lambda a, b: (a * b) % MOD}


@dataclass
class Expr:
    kind: str  # 'const' | 'ref' | 'op'
    value: int = 0  # const value
    pid: int = -1  # ref
    op: str = ""
    left: "Expr" = None
    right: "Expr" = None


def postorder_ops(e):
    """Op nodes of an expression in post-order (the root comes last)."""
    out = []

    def rec(n):
        if n.kind == "op":
            rec(n.left)
            rec(n.right)
            out.append(n)

    rec(e)
    return out


def count_ops(e):
    return len(postorder_ops(e))


def eval_expr(e, values):
    if e.kind == "const":
        return e.value
    if e.kind == "ref":
        return values[e.pid]
    return OPS[e.op](eval_expr(e.left, values), eval_expr(e.right, values))


@dataclass
class Param:
    pid: int
    kind: str  # 'inst' | 'abs'
    name: list  # words
    question: list  # words
    expr: Expr = None
    deps: list = field(default_factory=list)
    ftype: str = ""
    const: int = 0
    value: int = 0
    n_ops: int = 0
    name_ids: tuple = ()


@dataclass
class Problem:
    params: list
    query: int
    necessary: set
    order: list  # solution order (pids)
    var_letter: dict  # pid -> letter (str)
    temp_letters: dict  # pid -> list of temp letters (post-order, root excluded)
    statement: list  # words (sentences + question)
    solution: list  # words
    answer: int
    n_ops: int
    world: str
    prompt_ids: list = None  # <bos> statement question <sol>
    solution_ids: list = None  # solution ... <eos>

    @property
    def ids(self):
        return self.prompt_ids + self.solution_ids


@dataclass
class GenConfig:
    n_layers: int = 4
    items_per_layer: tuple = (2, 4)
    max_parents: int = 2
    max_instance_params: int = 20
    abs_depth: int = 2
    abs_ref_weight: float = 0.6  # relative weight of abstract params when choosing refs
    recent_bias: float = 0.5  # prob. of picking a ref among the most recently available params
    recent_k: int = 4
    ftype_weights: dict = field(default_factory=lambda: {
        "const": 0.20, "add_c": 0.12, "mul_c": 0.10, "sum": 0.14, "diff": 0.10,
        "prod": 0.06, "sum_add_c": 0.12, "mul_c_sum": 0.10, "mul_c_diff": 0.06})
    max_attempts: int = 200

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        g = cls()
        for k, v in d.items():
            if hasattr(g, k):
                setattr(g, k, tuple(v) if isinstance(getattr(g, k), tuple) else v)
        return g


N_REFS = {"const": 0, "add_c": 1, "mul_c": 1, "sum": 2, "diff": 2, "prod": 2,
          "sum_add_c": 2, "mul_c_sum": 2, "mul_c_diff": 2}


def _ref(pid):
    return Expr("ref", pid=pid)


def _const(c):
    return Expr("const", value=c)


def _op(op, a, b):
    return Expr("op", op=op, left=a, right=b)


def _build_formula(ftype, refs, c):
    if ftype == "const":
        return _const(c)
    if ftype == "add_c":
        return _op("+", _ref(refs[0]), _const(c))
    if ftype == "mul_c":
        return _op("*", _const(c), _ref(refs[0]))
    if ftype == "sum":
        return _op("+", _ref(refs[0]), _ref(refs[1]))
    if ftype == "diff":
        return _op("-", _ref(refs[0]), _ref(refs[1]))
    if ftype == "prod":
        return _op("*", _ref(refs[0]), _ref(refs[1]))
    if ftype == "sum_add_c":
        return _op("+", _op("+", _ref(refs[0]), _ref(refs[1])), _const(c))
    if ftype == "mul_c_sum":
        return _op("*", _const(c), _op("+", _ref(refs[0]), _ref(refs[1])))
    if ftype == "mul_c_diff":
        return _op("*", _const(c), _op("-", _ref(refs[0]), _ref(refs[1])))
    raise ValueError(ftype)


def _describe(ftype, c, refs, params):
    def ref(p):
        return ["each"] + params[p].name

    cs = str(c)
    if ftype == "const":
        return [cs]
    if ftype == "add_c":
        return [cs, "more", "than"] + ref(refs[0])
    if ftype == "mul_c":
        return [cs, "times"] + ref(refs[0])
    pair = ref(refs[0]) + ["and"] + ref(refs[1]) if len(refs) == 2 else None
    if ftype == "sum":
        return ["the", "sum", "of"] + pair
    if ftype == "diff":
        return ["the", "difference", "of"] + pair
    if ftype == "prod":
        return ["the", "product", "of"] + pair
    if ftype == "sum_add_c":
        return [cs, "more", "than", "the", "sum", "of"] + pair
    if ftype == "mul_c_sum":
        return [cs, "times", "the", "sum", "of"] + pair
    if ftype == "mul_c_diff":
        return [cs, "times", "the", "difference", "of"] + pair
    raise ValueError(ftype)


def _sum_chain(terms):
    e = terms[0]
    for t in terms[1:]:
        e = _op("+", e, t)
    return e


def _refs_of(e, out=None):
    out = [] if out is None else out
    if e.kind == "ref":
        out.append(e.pid)
    elif e.kind == "op":
        _refs_of(e.left, out)
        _refs_of(e.right, out)
    return out


def _build_graph(rng, cfg):
    world = rng.choice(WORLDS)
    layers = world["layers"][: cfg.n_layers]
    lo, hi = cfg.items_per_layer
    while True:
        items = [rng.sample(lay[1], rng.randint(lo, hi)) for lay in layers]
        edges = []  # (layer i, parent idx, child idx)
        for i in range(len(layers) - 1):
            for bi in range(len(items[i + 1])):
                k = rng.randint(1, min(cfg.max_parents, len(items[i])))
                for ai in rng.sample(range(len(items[i])), k):
                    edges.append((i, ai, bi))
        if len(edges) <= cfg.max_instance_params:
            break

    params = []
    inst_of = {}
    children = {}  # (layer, idx) -> list of (child idx, inst pid)
    for (i, ai, bi) in edges:
        a, b = items[i][ai], items[i + 1][bi]
        p = Param(pid=len(params), kind="inst", name=a.split() + ["'s"] + b.split(),
                  question=["How", "many"] + b.split() + ["does"] + a.split() + ["have", "?"])
        params.append(p)
        inst_of[(i, ai, bi)] = p.pid
        children.setdefault((i, ai), []).append((bi, p.pid))

    # Abstract parameters, bottom-up so that abs(b, j) exists before abs(a, j).
    abs_of = {}
    for i in range(len(layers) - 2, -1, -1):
        for ai in range(len(items[i])):
            kids = sorted(children.get((i, ai), []))
            for j in range(i + 1, min(len(layers), i + 1 + cfg.abs_depth)):
                if j == i + 1:
                    terms = [_ref(pid) for _, pid in kids]
                    if len(terms) < 2:
                        continue
                else:
                    if j != i + 2:
                        continue
                    terms = []
                    for bi, pid in kids:
                        gk = sorted(children.get((i + 1, bi), []))
                        if not gk:
                            continue
                        cnt = _ref(abs_of[(i + 1, bi, j)]) if len(gk) >= 2 else _ref(gk[0][1])
                        terms.append(_op("*", _ref(pid), cnt))
                    if not terms:
                        continue
                a = items[i][ai]
                cat = layers[j][0]
                p = Param(pid=len(params), kind="abs", name=a.split() + ["'s"] + cat.split(),
                          question=["How", "many"] + cat.split() + ["does"] + a.split() + ["have", "?"])
                p.expr = _sum_chain(terms)
                p.deps = sorted(set(_refs_of(p.expr)))
                p.ftype = "abs"
                params.append(p)
                abs_of[(i, ai, j)] = p.pid
    return world, params


def _assign_formulas(rng, cfg, params):
    inst = [p.pid for p in params if p.kind == "inst"]
    absp = [p.pid for p in params if p.kind == "abs"]
    rng.shuffle(inst)
    available, avail_set, topo = [], set(), []
    ftypes = list(cfg.ftype_weights.keys())
    fw = [cfg.ftype_weights[f] for f in ftypes]

    def add_available(pid):
        available.append(pid)
        avail_set.add(pid)
        topo.append(pid)
        changed = True
        while changed:
            changed = False
            for a in absp:
                if a not in avail_set and all(d in avail_set for d in params[a].deps):
                    available.append(a)
                    avail_set.add(a)
                    topo.append(a)
                    changed = True

    def pick_refs(n):
        chosen = []
        for _ in range(n):
            pool = [q for q in available if q not in chosen]
            if not pool:
                break
            if rng.random() < cfg.recent_bias:
                pool = pool[-cfg.recent_k:]
            w = [cfg.abs_ref_weight if params[q].kind == "abs" else 1.0 for q in pool]
            chosen.append(rng.choices(pool, weights=w)[0])
        return chosen

    for pid in inst:
        ft = rng.choices(ftypes, weights=fw)[0]
        n = N_REFS[ft]
        if len(available) < n:
            ft = rng.choice(["add_c", "mul_c"]) if available else "const"
            n = N_REFS[ft]
        refs = pick_refs(n)
        c = rng.randrange(MOD)
        if ft in ("mul_c", "mul_c_sum", "mul_c_diff"):
            c = rng.randrange(2, MOD)
        p = params[pid]
        p.ftype, p.const, p.expr, p.deps = ft, c, _build_formula(ft, refs, c), sorted(set(refs))
        p._refs = refs
        add_available(pid)
    assert len(topo) == len(params), "dependency graph is not complete"

    values = {}
    for pid in topo:
        values[pid] = eval_expr(params[pid].expr, values)
        params[pid].value = values[pid]
        params[pid].n_ops = count_ops(params[pid].expr)
    return topo


def _closures(params, topo):
    clos = {}
    for pid in topo:
        s = {pid}
        for d in params[pid].deps:
            s |= clos[d]
        clos[pid] = s
    return clos


def _random_topo(rng, params, nodes):
    nodes = set(nodes)
    indeg = {p: len([d for d in params[p].deps if d in nodes]) for p in nodes}
    users = {p: [] for p in nodes}
    for p in nodes:
        for d in params[p].deps:
            if d in nodes:
                users[d].append(p)
    ready = sorted(p for p in nodes if indeg[p] == 0)
    order = []
    while ready:
        p = ready.pop(rng.randrange(len(ready)))
        order.append(p)
        for u in users[p]:
            indeg[u] -= 1
            if indeg[u] == 0:
                ready.append(u)
    assert len(order) == len(nodes)
    return order


def render_step(p, letter, temps, params, var_letter):
    """Words of one solution line computing parameter p."""
    out = ["Define"] + p.name + ["as", letter, ";"]
    if p.expr.kind == "const":
        return out + ["so", letter, "=", str(p.expr.value), "."]
    ops = postorder_ops(p.expr)
    tmp = {id(n): temps[k] for k, n in enumerate(ops[:-1])}
    vals = {}

    def sym(n):
        if n.kind == "const":
            return str(n.value)
        if n.kind == "ref":
            return var_letter[n.pid]
        return tmp[id(n)]

    def val(n):
        if n.kind == "const":
            return n.value
        if n.kind == "ref":
            return params[n.pid].value
        return vals[id(n)]

    for k, n in enumerate(ops):
        r = OPS[n.op](val(n.left), val(n.right))
        vals[id(n)] = r
        lhs = ["so", letter] if k == len(ops) - 1 else [tmp[id(n)]]
        out += lhs + ["=", sym(n.left), n.op, sym(n.right), "=", str(val(n.left)), n.op,
                      str(val(n.right)), "=", str(r), "." if k == len(ops) - 1 else ";"]
    return out


def generate_problem(rng, cfg, op_range, tok=None):
    op_lo, op_hi = op_range
    target = rng.randint(op_lo, op_hi)
    fallback = None
    for attempt in range(cfg.max_attempts):
        world, params = _build_graph(rng, cfg)
        topo = _assign_formulas(rng, cfg, params)
        clos = _closures(params, topo)
        ops_of = {p: sum(params[q].n_ops for q in clos[p]) for p in clos}
        letters_of = {p: sum(max(1, params[q].n_ops) for q in clos[p]) for p in clos}
        ok = [p for p in clos if ops_of[p] == target and letters_of[p] <= len(LETTERS)]
        if ok:
            query = rng.choice(ok)
            break
        near = [p for p in clos if op_lo <= ops_of[p] <= op_hi and letters_of[p] <= len(LETTERS)]
        if near and fallback is None:
            fallback = (world, params, topo, clos, ops_of, rng.choice(near))
        if attempt > cfg.max_attempts // 2 and fallback is not None:
            world, params, topo, clos, ops_of, query = fallback
            break
    else:
        if fallback is None:
            raise RuntimeError(f"could not generate a problem with op in {op_range}")
        world, params, topo, clos, ops_of, query = fallback

    necessary = clos[query]
    order = _random_topo(rng, params, necessary)
    letters = rng.sample(LETTERS, sum(max(1, params[p].n_ops) for p in order))
    var_letter, temp_letters, li = {}, {}, 0
    for p in order:
        var_letter[p] = letters[li]
        temp_letters[p] = letters[li + 1: li + max(1, params[p].n_ops)]
        li += max(1, params[p].n_ops)

    # Statement: every instance parameter (necessary or distractor), shuffled.
    sents = []
    for p in params:
        if p.kind == "inst":
            sents.append(["The", "number", "of", "each"] + p.name + ["equals"]
                         + _describe(p.ftype, p.const, p._refs, params) + ["."])
    rng.shuffle(sents)
    statement = [w for s in sents for w in s] + ["Question", ":"] + params[query].question

    solution = []
    for p in order:
        solution += render_step(params[p], var_letter[p], temp_letters[p], params, var_letter)
    solution += ["Answer", ":", str(params[query].value), "."]

    prob = Problem(params=params, query=query, necessary=necessary, order=order,
                   var_letter=var_letter, temp_letters=temp_letters, statement=statement,
                   solution=solution, answer=params[query].value, n_ops=ops_of[query],
                   world=world["name"])
    if tok is not None:
        attach_ids(prob, tok)
    return prob


def attach_ids(prob, tok):
    prob.prompt_ids = [tok.bos] + tok.encode(prob.statement) + [tok.sol]
    prob.solution_ids = tok.encode(prob.solution) + [tok.eos]
    for p in prob.params:
        p.name_ids = tuple(tok.encode(p.name))
    return prob


class ProblemSampler:
    """Convenience wrapper: deterministic stream of problems for a seed."""

    def __init__(self, gen_cfg, op_range, seed, tok):
        self.rng = random.Random(seed)
        self.cfg = gen_cfg
        self.op_range = tuple(op_range)
        self.tok = tok

    def __call__(self, op_range=None):
        return generate_problem(self.rng, self.cfg, op_range or self.op_range, self.tok)


def make_test_set(gen_cfg, op_range, n, seed, tok):
    s = ProblemSampler(gen_cfg, op_range, seed, tok)
    return [s() for _ in range(n)]
