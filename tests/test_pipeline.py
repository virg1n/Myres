"""CPU tests: generator/checker consistency and metrics on a synthetic agent with known behaviour.

    python3 -m tests.test_pipeline
"""
import random

import numpy as np
import torch

from pilot import igsm
from pilot.evaluate import M, run_set
from pilot.metrics import summarize


def test_generator_and_checker(n=300):
    rng = random.Random(0)
    for _ in range(n):
        p = igsm.sample_problem(rng, "train", 2, 18)
        for style in ("forward", "backward"):
            words, err = igsm.solution_words(p, rng, style)
            c = igsm.check_solution(p, igsm.encode(words))
            assert c["correct"] and not c["has_error"] and err is None, (style, c)
        words, err = igsm.solution_words(p, rng, rng.choice(["forward", "backward"]), error=True)
        if words is None:
            continue
        c = igsm.check_solution(p, igsm.encode(words))
        assert c["has_error"] and c["first_err"] == err, (c["first_err"], err)
    print("generator/checker: ok")


class MockAgent:
    """Mode k solves correctly with prob acc[k]; errors are injected (natural-looking, propagated).
    Verifier j sees an error with signal strength 3.0 if j != i, 0.5 if j == i ("same blind spot")."""

    def __init__(self, probs, acc):
        self.by_prompt = {tuple(igsm.solve_prompt(p)): p for p in probs}
        self.by_tokens = {tuple(p.tokens()): p for p in probs}
        self.acc = acc
        self.sol_mode = {}

    def solve(self, prompts, mode, temp, seed):
        rng = random.Random(seed)
        out = []
        for pr in prompts:
            p = self.by_prompt[tuple(pr)]
            words = None
            if rng.random() > self.acc[mode]:
                words, _ = igsm.solution_words(p, rng, "forward", error=True)
            if words is None:
                words, _ = igsm.solution_words(p, rng, "forward")
            ids = igsm.encode(words)
            self.sol_mode[tuple(ids), p.seed if p.seed is not None else id(p)] = mode
            out.append(ids)
        return out

    def score(self, seqs, positions, mode):
        rng = np.random.default_rng(len(seqs) * 31 + mode)
        res = []
        for s, pos in zip(seqs, positions):
            n_prob = len(s) - 2
            sol_start = s.index(igsm.SOL) + 1
            p = self.by_tokens[tuple(s[2:sol_start - 1])]
            ids = s[sol_start:n_prob]
            c = igsm.check_solution(p, ids)
            solver = self.sol_mode.get((tuple(ids), p.seed if p.seed is not None else id(p)), mode)
            strength = 3.0 if solver != mode else 0.5
            logit = (strength if c["has_error"] else -strength) + rng.normal()
            lp = torch.full((2, igsm.VOCAB_SIZE), -50.0)
            lp[0, igsm.ERR], lp[0, igsm.OK] = logit / 2, -logit / 2
            step = c["first_err"] if c["has_error"] and solver != mode else 1
            lp[1, igsm.STEP_IDS[min(step, igsm.MAX_STEPS) - 1]] = 0.0
            res.append(lp)
        return res


def test_metrics():
    probs = igsm.fixed_set(3, "test", 400, 4, 10)
    agent = MockAgent(probs, acc=[0.7, 0.6, 0.65, 0.75])
    A, _ = run_set(agent, probs, 0.0, 1, 1024, None)
    m = summarize(A)
    print({k: round(v, 3) for k, v in m.items() if isinstance(v, float)})
    assert abs(m["pass1"] - 0.675) < 0.06, m["pass1"]
    assert m["ver_tpr_off"] > m["ver_tpr_diag"] + 0.3, (m["ver_tpr_off"], m["ver_tpr_diag"])
    assert m["auroc_off"] > 0.95 and 0.55 < m["auroc_diag"] < 0.85
    assert m["ver_fpr_off"] < 0.1 and m["ver_fpr_diag"] < 0.1
    assert m["ver_loc_off"] > 0.95 and m["ver_loc_diag"] < 0.3
    assert m["vvote_cross"] >= m["maj4"] - 0.02
    print("metrics on synthetic agent: ok")


if __name__ == "__main__":
    test_generator_and_checker()
    test_metrics()
