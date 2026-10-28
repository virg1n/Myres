import random

import pytest
import torch

from revise.examples import collate, erase, error_example, masked_window, runs_of, window_start
from revise.igsm.checker import ERR, OK, analyze, descendants
from revise.igsm.generator import GenConfig, generate_problem, make_test_set
from revise.igsm.tokenizer import Tokenizer
from revise.model import ModelConfig, RevisionLM, build_mask
from revise.utils import load_config

TOK = Tokenizer()
GEN = GenConfig()


def problems(n, op=(2, 15), seed=0):
    rng = random.Random(seed)
    return [generate_problem(rng, GEN, op, TOK) for _ in range(n)]


# ---------------------------------------------------------------- generator / checker

def test_generator_deterministic_and_in_range():
    a = make_test_set(GEN, (5, 9), 20, 7, TOK)
    b = make_test_set(GEN, (5, 9), 20, 7, TOK)
    assert [p.ids for p in a] == [p.ids for p in b]
    assert all(5 <= p.n_ops <= 9 for p in a)


def test_canonical_solutions_are_correct():
    for p in problems(200, (2, 20)):
        a = analyze(p, p.solution_ids, TOK)
        assert a.correct and a.complete and a.first_err is None
        assert all(s == OK for s in a.status)


def _value_positions(sol):
    """indices of final results ('= r .')"""
    return [k for k in range(1, len(sol) - 1)
            if sol[k] in TOK.num_of and sol[k - 1] == TOK["="] and sol[k + 1] == TOK["."]]


def test_consistent_error_propagation():
    """A wrong result followed by text that consistently uses it: exactly one
    root error, and the later values are its descendants."""
    for p in problems(50, (6, 12), seed=3):
        sol = list(p.solution_ids)
        vals = _value_positions(sol)
        # pick a step whose value is used later
        a0 = analyze(p, sol, TOK)
        cand = [j for j in vals if any(sol[k] in TOK.num_of for k in descendants(a0.parents, j))]
        if not cand:
            continue
        j = cand[0]
        wrong = (TOK.num_of[sol[j]] + 1) % 23
        bad = list(sol)
        bad[j] = TOK.num(wrong)
        a = analyze(p, bad, TOK)
        assert a.first_err == j and a.status[j] == ERR
        assert a.valid[j] == [sol[j]]
        d = descendants(a.parents, j)
        assert d and all(bad[k] in TOK.num_of for k in d), "value errors only propagate to values"
        return
    pytest.skip("no suitable problem")


def test_name_error_and_letter_reuse():
    p = problems(1, (6, 8), seed=5)[0]
    sol = list(p.solution_ids)
    # second step's variable letter reuses the first step's letter
    as_pos = [k for k, t in enumerate(sol) if t == TOK["as"]]
    bad = list(sol)
    bad[as_pos[1] + 1] = sol[as_pos[0] + 1]
    a = analyze(p, bad, TOK)
    assert a.first_err == as_pos[1] + 1
    assert sol[as_pos[0] + 1] not in a.valid[a.first_err]
    # name of the first step replaced by an item that does not exist
    bad = list(sol)
    bad[1] = TOK["Pencil"] if sol[1] != TOK["Pencil"] else TOK["Lion"]
    a = analyze(p, bad, TOK)
    assert a.first_err == 1


# ---------------------------------------------------------------- erasure / windows

def test_erase_tail_and_masks():
    sol = list(range(100, 120))
    new, info = erase(sol, 3, [5, 6, 10, 18, 19], mask_id=0, slack=2, slack_min_len=2)
    assert len(new) == 18 - 0 + 2  # tail (18, 19) truncated; run 5-6 gets 2 slack masks
    assert new[:5] == sol[:5] and new[5:9] == [0, 0, 0, 0] and new[9:12] == sol[7:10]
    assert runs_of([1, 2, 3, 7, 9, 10]) == [(1, 3), (7, 7), (9, 10)]
    assert window_start([1, 2, 0, 3, 4, 5], 3, 0) == 2
    assert window_start([1, 2, 3, 4, 5], 3, 0) == 2


def test_masked_window_targets_reconstruct_solution():
    cfg = load_config("configs/small.yaml")["revision"]
    rng = random.Random(0)
    n = 0
    for p in problems(40, (4, 10), seed=11):
        a = analyze(p, p.solution_ids, TOK)
        w = masked_window(p, a.parents, a.lines, cfg["window"], rng, TOK, cfg)
        if w is None:
            continue
        P = len(p.prompt_ids)
        filled = [f[0] for f in w["fix"] if f[0] != TOK.null]
        start = w["start"] - P
        assert p.solution_ids[start:start + len(filled)] == filled
        assert TOK.mask in w["ids"]
        n += 1
    assert n > 10


# ---------------------------------------------------------------- model / masks

def test_window_copy_cannot_see_future():
    torch.manual_seed(0)
    m = RevisionLM(ModelConfig(vocab_size=len(TOK), d_model=32, n_layers=2, n_heads=4, d_ff=64)).eval()
    # causal stream of 12 tokens; window copy of causal positions 4..7 (written so far = 8 tokens)
    causal = torch.randint(10, 100, (1, 12))
    copy = causal[:, 4:8].clone()
    ids = torch.cat([causal, copy], 1)
    pos = torch.cat([torch.arange(12), torch.arange(4, 8)])[None]
    is_copy = torch.tensor([[False] * 12 + [True] * 4])
    wstart = torch.tensor([[0] * 12 + [4] * 4])
    wid = torch.tensor([[-1] * 12 + [0] * 4])
    valid = torch.ones_like(is_copy)
    mask = build_mask(pos, is_copy, wstart, wid, valid)
    h1 = m(ids, pos, is_copy.long(), mask)
    ids2 = ids.clone()
    ids2[0, 4:12] = 7  # change the causal tokens from the window start on: copy must not care
    h2 = m(ids2, pos, is_copy.long(), mask)
    assert torch.allclose(h1[0, 12:], h2[0, 12:], atol=1e-5)
    ids3 = ids.clone()
    ids3[0, 2] = 7  # prefix token before the window: copy must see it
    h3 = m(ids3, pos, is_copy.long(), mask)
    assert not torch.allclose(h1[0, 12:], h3[0, 12:], atol=1e-5)
    # causal stream never sees the copy
    ids4 = ids.clone()
    ids4[0, 12:] = 7
    h4 = m(ids4, pos, is_copy.long(), mask)
    assert torch.allclose(h1[0, :12], h4[0, :12], atol=1e-5)


def test_training_losses_and_confdrop_run():
    from revise.train import compute_losses, confdrop_targets, forward_batch
    from revise.examples import correct_example
    cfg = load_config("configs/tiny.yaml")
    torch.manual_seed(0)
    m = RevisionLM(ModelConfig.from_dict(cfg["model"], len(TOK)))
    rng = random.Random(0)
    exs = []
    for p in problems(4, (3, 6), seed=2):
        a = analyze(p, p.solution_ids, TOK)
        exs.append(correct_example(p, a, rng, TOK, cfg["revision"], 1, 1))
        # synthetic "own error": corrupt a value, keep the rest consistent-ish
        sol = list(p.solution_ids)
        j = _value_positions(sol)[0]
        sol[j] = TOK.num((TOK.num_of[sol[j]] + 5) % 23)
        an = analyze(p, sol, TOK)
        if an.first_err is None:
            continue
        e = {"prompt": p.prompt_ids, "sol": sol, "status": an.status, "valid": an.valid,
             "parents": an.parents, "j": an.first_err}
        exs.append(error_example(e, rng, TOK, cfg["revision"]))
    batch = collate(exs, TOK.pad)
    h = forward_batch(m, batch, None)
    n = confdrop_targets(m, h, batch, cfg, None, TOK.mask)
    assert n > 0 and (batch["dep_t"] >= 0).any()
    loss, metrics, _ = compute_losses(m, h, batch, cfg, 0, "revise")
    loss.backward()
    assert torch.isfinite(loss)
    assert m.err_head.weight.grad is not None and m.dep_q.weight.grad is not None
