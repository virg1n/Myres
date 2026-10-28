"""Solution checker / labeller.

Given a problem (with its true dependency graph) and a solution token sequence
(possibly partial, possibly written by a model with mistakes), `analyze` walks
the solution left to right and returns, for every solution token:

* status:  OK (consistent with what is written before it), ERR (a *root* error:
  the first token that is inconsistent with the problem or with the text before
  it), or UNK (cannot be judged, e.g. after the parser lost alignment).
  Labels follow *consistency* semantics: a value that was correctly computed
  from an earlier wrong value is OK -- it is not a new error, it *depends* on
  the error. Fixing the root error makes such tokens inconsistent.
* valid:   for ERR tokens, the set of token ids that would be correct here
  (used as the target of the fix head).
* parents: the information-flow DAG between solution tokens, derived from the
  generator's dependency graph (which value was computed from which operands,
  which formula belongs to which parameter, which variable letter refers to
  which definition, ...). `descendants(j)` = all later tokens that depend on
  token j. This is the ground truth used by the oracle erasure variant.
"""

from dataclasses import dataclass

from .generator import OPS, postorder_ops

OK, ERR, UNK = 0, 1, -1
ALL = None  # parents == ALL: the token depends on every earlier token (parser lost alignment)


@dataclass
class Analysis:
    status: list
    valid: list
    parents: list
    first_err: int
    answer: int
    complete: bool
    correct: bool
    lines: list
    n_steps: int

    def descendants(self, j):
        return descendants(self.parents, j)

    @property
    def n_errors(self):
        return sum(1 for s in self.status if s == ERR)


def descendants(parents, j):
    n = len(parents)
    children = [[] for _ in range(n)]
    for k in range(n):
        if parents[k] is not ALL:
            for p in parents[k]:
                if 0 <= p < n:
                    children[p].append(k)
    seen = set()
    stack = [j]
    while stack:
        u = stack.pop()
        for c in children[u]:
            if c not in seen:
                seen.add(c)
                stack.append(c)
    for k in range(j + 1, n):
        if parents[k] is ALL:
            seen.add(k)
    return sorted(k for k in seen if k > j)


def line_ids(sol, tok):
    dot = tok["."]
    out, cur = [], 0
    for t in sol:
        out.append(cur)
        if t == dot:
            cur += 1
    return out


class _Lost(Exception):
    pass


class _End(Exception):
    pass


def analyze(prob, sol, tok, strict_necessary=True):
    n = len(sol)
    status = [UNK] * n
    valid = [None] * n
    parents = [ALL] * n
    st = {"k": 0, "first_err": None}

    T = {w: tok[w] for w in ["Define", "Answer", "as", ";", "so", "=", ".", ":", "+", "-", "*"]}
    op_tok = {"+": T["+"], "-": T["-"], "*": T["*"]}
    tok_op = {v: k for k, v in op_tok.items()}
    letters_all = tok.letter_ids
    letter_set = tok.letter_set
    num_of = tok.num_of
    params = prob.params
    name_to_pid = {p.name_ids: p.pid for p in params}

    defined = {}  # pid -> dict(name_idx, letter, letter_idx, value, value_idx)
    binding = {}  # letter id -> dict(value, value_idx, decl_idx)
    used = set()

    def peek():
        if st["k"] >= n:
            raise _End
        return sol[st["k"]]

    def emit(ok, vset, pars, unknown=False):
        k = st["k"]
        if unknown:
            status[k] = UNK
        elif ok:
            status[k] = OK
        else:
            status[k] = ERR
            valid[k] = sorted(vset)
            if st["first_err"] is None:
                st["first_err"] = k
        parents[k] = list(pars)
        st["k"] = k + 1
        return k

    def structural(word, pars):
        t = peek()
        ok = t == T[word]
        k = emit(ok, {T[word]}, pars)
        if not ok:
            raise _Lost
        return k

    def letter_slot(expected, pars):
        """Slot that must contain a specific (already known) letter."""
        t = peek()
        if t not in letter_set and t not in num_of:
            emit(False, {expected}, pars)
            raise _Lost
        return emit(t == expected, {expected}, pars), t

    def symbol(child, name_idx, temps):
        """Operand symbol: a constant, a variable letter, or a temp letter."""
        t = peek()
        if child.kind == "const":
            exp, pars = tok.num(child.value), list(name_idx)
        elif child.kind == "ref":
            d = defined[child.pid]
            exp, pars = d["letter"], list(name_idx) + [d["letter_idx"]]
        else:
            ti = temps[id(child)]
            exp, pars = ti["letter"], list(name_idx) + [ti["letter_idx"]]
        if t not in letter_set and t not in num_of:
            emit(False, {exp}, pars)
            raise _Lost
        k = emit(t == exp, {exp}, pars)
        if t in num_of:
            return {"idx": k, "value": num_of[t], "src": None}
        b = binding.get(t)
        if b is None:
            return {"idx": k, "value": None, "src": None}
        return {"idx": k, "value": b["value"], "src": b["value_idx"]}

    def value_slot(sym):
        t = peek()
        pars = [sym["idx"]] + ([sym["src"]] if sym["src"] is not None else [])
        if t not in num_of:
            emit(False, {tok.num(sym["value"])} if sym["value"] is not None else set(), pars)
            raise _Lost
        if sym["value"] is None:
            k = emit(True, None, pars, unknown=True)
        else:
            k = emit(num_of[t] == sym["value"], {tok.num(sym["value"])}, pars)
        return {"idx": k, "value": num_of[t]}

    def op_slot(expected_tok, pars):
        t = peek()
        if t not in tok_op:
            emit(False, {expected_tok}, pars)
            raise _Lost
        return emit(t == expected_tok, {expected_tok}, pars), t

    def parse_step(define_idx):
        query_ready = None
        cands = [p for p in params if p.pid not in defined and all(d in defined for d in p.deps)
                 and (p.pid in prob.necessary or not strict_necessary)]
        written, name_idx, diverged = [], [], False
        while True:
            t = peek()
            o = len(written)
            nxt = set()
            for p in cands:
                if p.name_ids[:o] == tuple(written):
                    nxt.add(p.name_ids[o] if o < len(p.name_ids) else T["as"])
            if t == T["as"] and o > 0:
                break
            if not diverged:
                k = emit(t in nxt, nxt, name_idx)
                diverged = t not in nxt
            else:
                pref = tuple(written + [t])
                match = any(p.name_ids[: o + 1] == pref for p in params)
                k = emit(True, None, name_idx, unknown=not match)
            written.append(t)
            name_idx.append(k)
            if len(written) > 16:
                raise _Lost
        wpid = name_to_pid.get(tuple(written))
        if not diverged:
            emit(T["as"] in nxt, nxt, name_idx)
            diverged = T["as"] not in nxt
        else:
            emit(True, None, name_idx, unknown=wpid is None)
        if wpid is None:
            raise _Lost
        wp = params[wpid]
        dep_names = []
        for d in wp.deps:
            if d in defined:
                dep_names += defined[d]["name_idx"]
        for i, k in enumerate(name_idx):
            parents[k] = name_idx[:i] + dep_names
        if not all(d in defined for d in wp.deps):
            raise _Lost  # formula refers to undefined quantities; cannot align further

        # variable letter
        t = peek()
        free = [c for c in letters_all if c not in used]
        if t not in letter_set:
            emit(False, free, name_idx)
            raise _Lost
        letter_idx = emit(t not in used, free, name_idx)
        letter = t
        used.add(t)
        binding[letter] = {"value": None, "value_idx": None, "decl_idx": letter_idx}
        structural(";", name_idx)

        e = wp.expr
        if e.kind == "const":
            structural("so", name_idx)
            letter_slot(letter, name_idx + [letter_idx])
            structural("=", name_idx)
            t = peek()
            if t not in num_of:
                emit(False, {tok.num(e.value)}, name_idx)
                raise _Lost
            vidx = emit(num_of[t] == e.value, {tok.num(e.value)}, name_idx)
            value = num_of[t]
            structural(".", name_idx)
        else:
            ops = postorder_ops(e)
            temps = {}
            for ci, node in enumerate(ops):
                root = ci == len(ops) - 1
                if root:
                    structural("so", name_idx)
                    letter_slot(letter, name_idx + [letter_idx])
                    lhs_letter, lhs_idx = letter, letter_idx
                else:
                    t = peek()
                    free = [c for c in letters_all if c not in used]
                    if t not in letter_set:
                        emit(False, free, name_idx)
                        raise _Lost
                    lhs_idx = emit(t not in used, free, name_idx)
                    lhs_letter = t
                    used.add(t)
                structural("=", name_idx)
                sa = symbol(node.left, name_idx, temps)
                op_idx, op_written = op_slot(op_tok[node.op], name_idx)
                sb = symbol(node.right, name_idx, temps)
                structural("=", name_idx)
                va = value_slot(sa)
                opv_idx, opv = op_slot(op_written, [op_idx])
                vb = value_slot(sb)
                structural("=", name_idx)
                t = peek()
                pars = [va["idx"], opv_idx, vb["idx"]]
                if t not in num_of:
                    emit(False, set(), pars)
                    raise _Lost
                res = OPS[tok_op[opv]](va["value"], vb["value"])
                ridx = emit(num_of[t] == res, {tok.num(res)}, pars)
                value, vidx = num_of[t], ridx
                temps[id(node)] = {"letter": lhs_letter, "letter_idx": lhs_idx}
                if lhs_letter in binding and binding[lhs_letter]["value_idx"] is None:
                    binding[lhs_letter]["value"] = value
                    binding[lhs_letter]["value_idx"] = vidx
                elif lhs_letter not in binding:
                    binding[lhs_letter] = {"value": value, "value_idx": vidx, "decl_idx": lhs_idx}
                structural("." if root else ";", name_idx)
        binding[letter]["value"] = value
        binding[letter]["value_idx"] = vidx
        if wpid not in defined:
            defined[wpid] = {"name_idx": name_idx, "letter": letter, "letter_idx": letter_idx,
                             "value": value, "value_idx": vidx}
        return query_ready

    answer, complete, n_steps = None, False, 0
    try:
        while True:
            t = peek()
            qdone = prob.query in defined
            if t == T["Answer"]:
                emit(qdone, {T["Define"]}, [])
                structural(":", [])
                q = defined.get(prob.query)
                t = peek()
                pars = [q["value_idx"]] if q is not None else []
                if t not in num_of:
                    emit(False, {tok.num(q["value"])} if q else set(), pars)
                    raise _Lost
                if q is None:
                    emit(True, None, pars, unknown=True)
                else:
                    emit(num_of[t] == q["value"], {tok.num(q["value"])}, pars)
                answer = num_of[t]
                structural(".", [])
                t = peek()
                emit(t == tok.eos, {tok.eos}, [])
                complete = t == tok.eos
                if not complete:
                    raise _Lost
                break
            elif t == T["Define"]:
                k = emit(not qdone, {T["Answer"]}, [])
                n_steps += 1
                parse_step(k)
            else:
                emit(False, {T["Answer"] if qdone else T["Define"]}, [])
                raise _Lost
    except (_Lost, _End):
        pass
    # tokens after a loss of alignment keep parents == ALL and status UNK.
    correct = answer is not None and answer == prob.answer
    return Analysis(status=status, valid=valid, parents=parents, first_err=st["first_err"],
                    answer=answer, complete=complete, correct=correct, lines=line_ids(sol, tok),
                    n_steps=n_steps)


def parse_answer(sol, tok):
    """Final answer from a solution (last 'Answer : N'), without full checking."""
    a, c = tok["Answer"], tok[":"]
    for i in range(len(sol) - 2, -1, -1):
        if sol[i] == a and i + 2 < len(sol) and sol[i + 1] == c and sol[i + 2] in tok.num_of:
            return tok.num_of[sol[i + 2]]
    return None
