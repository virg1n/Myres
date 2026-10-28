"""The confidence-drop measurement must recover exactly the tokens that relied
on an error, given a writer that is confident in consistent text.

A realistic model mistake is built by computing one step's result wrongly and
writing every later step *consistently* with the wrong value. A scripted
"perfect writer" (log p = 0 for the token that is consistent with the text
before it, -10 otherwise) is plugged in place of the network. After fixing the
error, the measured dependents must equal the tokens that differ between the
mistaken and the correct solution (plus, possibly, dependents whose value
happens to come out the same), and nothing outside the true dependency graph."""

import copy
import random

import torch

import revise.decode as D
from revise.examples import collate, error_example
from revise.igsm.checker import ERR, analyze
from revise.igsm.generator import GenConfig, eval_expr, generate_problem, render_step
from revise.igsm.tokenizer import Tokenizer
from revise.train import confdrop_targets, forward_batch
from revise.utils import load_config

TOK = Tokenizer()


def consistent_mistake(p, target, delta=5):
    q = copy.deepcopy(p)
    vals = {}
    for pid in q.order:
        v = eval_expr(q.params[pid].expr, vals)
        if pid == target:
            v = (v + delta) % 23
        vals[pid] = v
        q.params[pid].value = v
    words = []
    for pid in q.order:
        w = render_step(q.params[pid], q.var_letter[pid], q.temp_letters[pid], q.params, q.var_letter)
        if pid == target:
            w[len(w) - 2] = str(vals[pid])  # the step's own (wrong) result
        words += w
    words += ["Answer", ":", str(vals[q.query]), "."]
    return TOK.encode(words) + [TOK.eos]


class PerfectWriter(torch.nn.Module):
    def __init__(self, problems):
        super().__init__()
        self.by_prompt = {tuple(p.prompt_ids): p for p in problems}

    def forward(self, ids, pos, stream, mask, cache=None):
        B, L = ids.shape
        out = torch.full((B, L, len(TOK)), -10.0)
        for b in range(B):
            row = ids[b].tolist()
            s = row.index(TOK.sol)
            n = int(((stream[b] == 0) & (ids[b] != TOK.pad)).sum())
            p = self.by_prompt[tuple(row[: s + 1])]
            sol = row[s + 1: n]
            a = analyze(p, sol, TOK)
            for k in range(len(sol)):
                fav = a.valid[k][0] if a.status[k] == ERR and a.valid[k] else sol[k]
                out[b, s + k, fav] = 0.0
        return out

    def lm_logits(self, h):
        return h

    def fix_logits(self, h):
        return h


def _cases(n=30, seed=0):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        p = generate_problem(rng, GenConfig(), (6, 10), TOK)
        target = p.order[rng.randrange(len(p.order) - 1)]
        wrong = consistent_mistake(p, target)
        a = analyze(p, wrong, TOK)
        if a.first_err is None:
            continue
        j = a.first_err
        assert a.n_errors == 1, "later text must be consistent with the mistake"
        changed = {k for k in range(j + 1, len(wrong)) if wrong[k] != p.solution_ids[k]}
        deps = set(a.descendants(j))
        assert changed <= deps
        out.append((p, wrong, a, j, (changed, deps)))
    return out


def test_training_confdrop_labels_match_true_dependents():
    cfg = load_config("configs/small.yaml")
    cfg["revision"]["confdrop_iters"] = 30
    cases = _cases()
    model = PerfectWriter([c[0] for c in cases])
    rng = random.Random(1)
    exs, meta = [], []
    for p, wrong, a, j, truth in cases:
        e = {"prompt": p.prompt_ids, "sol": wrong, "status": a.status, "valid": a.valid, "parents": a.parents, "j": j}
        ex = error_example(e, rng, TOK, {**cfg["revision"], "neg_window_prob": 0.0})
        w = ex["windows"][0]
        ws = w["start"] - len(p.prompt_ids)
        meta.append((ws, len(w["ids"]), truth))
        exs.append(ex)
    batch = collate(exs, TOK.pad)
    h = forward_batch(model, batch, None)
    confdrop_targets(model, h, batch, cfg, None, TOK.mask)
    n_pos = 0
    for r, (row, b, off, n, jw, ft) in enumerate(batch["confdrop"]):
        ws, m, truth = meta[b]
        got = {ws + k for k in range(n) if batch["dep_t"][row, k] == 1}
        changed, deps = truth
        # every token that changes is found; nothing outside the true dependency graph is flagged
        assert {k for k in changed if k < ws + m} <= got <= deps, (b, sorted(got), sorted(changed))
        n_pos += len(got)
    assert n_pos > 0


def test_decoder_confdrop_matches_true_dependents():
    cases = _cases(10, seed=3)
    model = PerfectWriter([c[0] for c in cases])
    runner = D.Runner(model, TOK, torch.device("cpu"))
    pol = D.Policy(erase="confdrop", confdrop_iters=50)
    for p, wrong, a, j, truth in cases:
        slot = D.Slot(idx=0, prob=p, sol=list(wrong))
        P = len(p.prompt_ids)
        out = runner.forward([(p.prompt_ids + wrong, None, 0)])
        new = p.solution_ids[j]
        deps = D._measure_confdrop(runner, [slot], [(0, 0, j, new)], out, pol, TOK, [(P, 0, True)])
        changed, true_deps = truth
        assert changed <= set(deps[0]) <= true_deps
