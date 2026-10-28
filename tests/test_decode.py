"""End-to-end test of the revision decoding loop with a scripted fake model:
it writes the reference solution but makes one arithmetic mistake, notices it
only ~20 tokens later, fixes it, and fills erased tokens with the reference."""

import random

import torch

import revise.decode as D
from revise.igsm.checker import analyze, descendants
from revise.igsm.generator import GenConfig, generate_problem
from revise.igsm.tokenizer import Tokenizer
from revise.model import ModelConfig, RevisionLM

TOK = Tokenizer()


def _setup(seed=4):
    rng = random.Random(seed)
    while True:
        p = generate_problem(rng, GenConfig(), (8, 12), TOK)
        sol = p.solution_ids
        an = analyze(p, sol, TOK)
        vals = [k for k in range(1, len(sol) - 1)
                if sol[k] in TOK.num_of and sol[k - 1] == TOK["="] and sol[k + 1] == TOK["."]]
        for j in vals:
            d = descendants(an.parents, j)
            if len([k for k in d if k < j + 20]) >= 2 and j + 40 < len(sol):
                return p, j


class FakeRunner(D.Runner):
    def __init__(self, model, tok, device, prob, j0):
        super().__init__(model, tok, device)
        self.prob, self.j0 = prob, j0
        self.ref = prob.solution_ids
        self.wrong = TOK.num((TOK.num_of[self.ref[j0]] + 3) % 23)

    def forward(self, items):
        B = len(items)
        V = len(TOK)
        P = len(self.prob.prompt_ids)
        Wm = max(len(w) if w else 0 for _, w, _ in items) or 1
        lm = torch.full((B, V), -10.0)
        err = torch.zeros(B, Wm)
        fill = torch.zeros(B, Wm, dtype=torch.long)
        fix = torch.zeros(B, Wm, dtype=torch.long)
        cvalid = torch.zeros(B, Wm, dtype=torch.bool)
        ctok = torch.full((B, Wm), TOK.pad)
        for b, (c, w, s) in enumerate(items):
            sol = c[P:]
            n = len(sol)
            nxt = self.wrong if n == self.j0 else (self.ref[n] if n < len(self.ref) else TOK.eos)
            lm[b, nxt] = 10.0
            if w:
                ws = s - P
                for i, t in enumerate(w):
                    k = ws + i
                    cvalid[b, i] = True
                    ctok[b, i] = t
                    fill[b, i] = self.ref[k]
                    fix[b, i] = self.ref[k]
                    if k == self.j0 and t == self.wrong and n >= self.j0 + 20:
                        err[b, i] = 0.99
        return {"lm": lm, "err": err, "fill": fill, "fix": fix, "cvalid": cvalid, "ctok": ctok,
                "disagree": fix != ctok,
                "clen": [len(w) if w else 0 for _, w, _ in items], "coff": [len(c) for c, _, _ in items]}


def _run(policy_name, monkeypatch):
    p, j0 = _setup()
    model = RevisionLM(ModelConfig(vocab_size=len(TOK), d_model=16, n_layers=1, n_heads=2, d_ff=32))
    monkeypatch.setattr(D, "Runner", lambda m, tok, dev, amp=None: FakeRunner(m, tok, dev, p, j0))
    pol = D.make_policy(policy_name, {"window": 64, "max_sol_len": 800, "slack_min_len": 99})
    res = D.decode_revision(model, [p], pol, TOK, torch.device("cpu"), batch_size=1)[0]
    return p, j0, res


def test_oracle_erasure_fixes_and_refills(monkeypatch):
    p, j0, r = _run("oracle_dep", monkeypatch)
    assert r["correct"] and r["sol_correct"]
    assert r["fixes"] == 1 and r["erased"] >= 2 and r["fills"] >= 1
    ev = r["events"][0]
    assert ev["j"] == j0 and ev["true_err"] and ev["fix_valid"]
    assert ev["tp"] == ev["n_erase"] == ev["n_true_dep"]


def test_suffix_erasure_rewrites_tail(monkeypatch):
    p, j0, r = _run("erase_suffix", monkeypatch)
    assert r["correct"] and r["fixes"] == 1 and r["fills"] == 0
    assert r["erased"] >= 19


def test_fix_only(monkeypatch):
    p, j0, r = _run("fix_only", monkeypatch)
    assert r["fixes"] == 1 and r["erased"] == 0 and r["correct"]
