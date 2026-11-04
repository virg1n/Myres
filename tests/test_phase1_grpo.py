"""GRPO with and without modes on CPU with a tiny random Qwen3 and a fabricated phase1_self (~5-10 min).

    PYTHONUTF8=1 python3 -m tests.test_phase1_grpo

1. advantages and the loss on hand-computed numbers: Dr. GRPO, groups of equal rewards, the coverage ablation,
   the normalizer; the reward rule; problem order, checkpoints, steps from the calibration
2. the gradient of the loss against a row-by-row computation (masks, the off-by-one of the targets, padding,
   modes), for FiLM codes, separate adapters and no modes
3. routing: a rollout of mode k changes only code k / adapter k (and the shared LoRA of FiLM)
4. rollouts: left padding, mixed modes and the token budget per call give the tokens of single rows, the cut after the
   stop token, sampling at T = 1 without top-k / top-p although generation_config.json of the base says otherwise
5. the trainer: rollout accounting, the same problems for every chain, equal rewards train nothing, checkpoints in the
   adapter format of phase1.train, a stopped and resumed run equals an uninterrupted one, the safety time limit keeps
   last.pt, a changed setting or a sampling other than pi refused, calibration and pinned steps, the easy-problem filter
6. plan: jobs, dependencies, simulated time, calibration or fixed steps, refused inputs (M_0 of another max_tokens or
   adapter version) and settings
7. the --smoke run end to end through the real job runner (HF backend), the report, the comparison with RFT"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zlib
from types import SimpleNamespace as NS

import numpy as np
import torch

from phase1 import film as FM
from phase1 import greport, grun
from phase1 import grpo as G
from tests.test_phase1_seeds import ENV, tiny_model
from tests.test_phase1_self import fake_gen, problems, write_jsonl
from tests.test_phase1_z import fake_refs

K = 4


def close(a, b, tol=1e-9):
    return np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), atol=tol, rtol=0)


def test_math():
    assert close(G.group_adv([1, 0, 0, 1]), [0.5, -0.5, -0.5, 0.5])
    assert close(G.group_adv([1, 0, 0, 0]), [0.75, -0.25, -0.25, -0.25])
    for r in ([0] * 8, [1] * 8):
        A = G.group_adv(r, [k // 2 for k in range(8)], K, "cover")
        assert close(A, 0) and not np.isnan(A).any(), "equal rewards: no gradient, no division"
    # cover: modes 0..3 with 2 rollouts each, p = (0.5, 0, 1, 0); A = r - 0.375 scaled by prod_{j != k} (1 - p_j)
    # weights (1 - p_j) over the other modes: mode 0: 1 * 0 * 1, mode 1: 0.5 * 0 * 1, mode 2: 0.5 * 1 * 1, mode 3: 0
    r, m = [1, 0, 0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 2, 2, 3, 3]
    assert close(G.group_adv(r, m, K), [0.625, -0.375, -0.375, -0.375, 0.625, 0.625, -0.375, -0.375])
    assert close(G.group_adv(r, m, K, "cover"), [0, 0, 0, 0, 0.3125, 0.3125, 0, 0])
    r = [0, 1, 0, 0, 0, 0, 0, 0]                 # only mode 0 solves: its success is exclusive, full weight
    assert close(G.group_adv(r, m, K, "cover"), [-0.125, 0.875, -0.125 * 0.5, -0.125 * 0.5, -0.125 * 0.5,
                                                 -0.125 * 0.5, -0.125 * 0.5, -0.125 * 0.5])
    # the loss L = sum_i A_i NLL_i / (P G max_tokens): P = 1, G = 4, max_tokens 10, NLL (2, 3, 5, 7)
    c = G.pg_coef(G.group_adv([1, 0, 0, 0]), 10)
    assert close(c, [0.01875, -0.00625, -0.00625, -0.00625])
    assert abs(float(c @ np.array([2, 3, 5, 7])) - (0.75 * 2 - 0.25 * 15) / 40) < 1e-12
    assert G.reward("so \\boxed{7}", "7") == 1 and G.reward("so \\boxed{7.0}", "7") == 1
    assert G.reward("the answer is 7", "7") == 0, "a right answer without \\boxed{} is not admitted"
    assert G.reward("\\boxed{8}", "7") == 0 and G.reward("", "7") == 0
    n, P = 10, 4
    seq = [i for t in range(5) for i in G.step_problems(n, 3, P, t)]
    assert sorted(seq[:n]) == list(range(n)) and sorted(seq[n:2 * n]) == list(range(n)), "every epoch covers the pool"
    assert seq[:n] != seq[n:2 * n] and seq == [i for t in range(5) for i in G.step_problems(n, 3, P, t)]
    assert G.step_problems(n, 3, P, 2) != G.step_problems(n, 4, P, 2), "the seed sets the order"
    assert G.ckpt_steps(300, 3) == [100, 200, 300] and G.ckpt_steps(2, 2) == [1, 2]
    cal = [{"min_per_step": 1.5}, {"min_per_step": 2.0}]           # the slowest: 9 h * 60 / (1.1 * 2) = 245.4
    assert G.plan_steps(cal, 9, 3, 1000) == 243 and G.plan_steps(cal, 9, 3, 100) == 99 and G.plan_steps(cal, 0.01, 3, 9) == 3
    print("advantages, loss, reward, order, steps: ok")


def build(base, kind):
    """A tiny policy with non-trivial modes: B of LoRA and the FiLM output layer random."""
    torch.manual_seed(1)
    model, film, ctx = FM.build(base, K if kind == "film" else 0, 4, 8, "cpu", grad_ckpt=False,
                                lora_modes=K if kind == "mcl" else 0)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith(".B"):
                p.normal_(0, 0.3)
        if film is not None:
            film.net[2].weight.normal_(0, 0.3)
    lora, fps = FM.trainable(model, film)
    return model, film, ctx, lora + fps


def rand_rows(n, modes, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [{"p": torch.randint(1, 300, (int(torch.randint(3, 9, (1,), generator=g)),), generator=g).tolist(),
             "t": torch.randint(1, 300, (int(torch.randint(1, 7, (1,), generator=g)),), generator=g).tolist(),
             "mode": modes[i]} for i in range(n)]


def manual_loss(model, film, ctx, rows, coef):
    """sum_i coef_i * (- sum_t log p(t_t | p, t_<t)), one row at a time without padding."""
    tot, nll = 0.0, []
    for r, c in zip(rows, coef):
        ctx.gamma = film.gamma(torch.tensor([r["mode"]])) if film is not None and r["mode"] is not None else None
        ctx.modes = torch.tensor([r["mode"]]) if r["mode"] is not None else None
        x = torch.tensor([r["p"] + r["t"]])
        lp = torch.log_softmax(model(input_ids=x).logits[0].float(), -1)
        L = len(r["p"])
        v = -sum(lp[L - 1 + i, tok] for i, tok in enumerate(r["t"]))
        nll.append(float(v.detach()))
        tot = tot + c * v
    tot.backward()
    ctx.gamma, ctx.modes = None, None
    return nll


def test_loss(base):
    for kind in ("film", "mcl", "none"):
        model, film, ctx, ps = build(base, kind)
        rows = rand_rows(7, [None] * 7 if kind == "none" else [0, 2, 1, 2, 3, 0, 2])
        coef = [0.3, -0.1, 0.2, 0.05, -0.4, 0.15, -0.25]
        nll, ntok = G.backward_pg(model, film, ctx, rows, coef, 0, "cpu", mb_tokens=40)      # padded micro-batches
        g1 = [p.grad.clone() for p in ps]
        for p in ps:
            p.grad = None
        ref = manual_loss(model, film, ctx, rows, coef)
        assert close(nll, ref, 1e-4) and ntok == [len(r["t"]) for r in rows], (kind, nll, ref)
        for a, b in zip(g1, [p.grad for p in ps]):          # float32 round-off: ~1e-6 of the largest entry
            assert float((a - b).abs().max()) <= 2e-5 * float(b.abs().max()) + 1e-8, (kind, float((a - b).abs().max()))
        assert ctx.gamma is None and ctx.modes is None
    print("gradient of the loss = row-by-row computation (FiLM codes, separate adapters, no modes): ok")


def test_routing(base):
    for kind in ("film", "mcl", "none"):
        model, film, ctx, ps = build(base, kind)
        before = {**{n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad},
                  **({f"film.{n}": p.detach().clone() for n, p in film.named_parameters()} if film else {})}
        opt = torch.optim.AdamW(ps, lr=1e-2, weight_decay=0.0)
        mode = None if kind == "none" else 2
        G.backward_pg(model, film, ctx, rand_rows(3, [mode] * 3), [0.5, -0.2, 0.3], 0, "cpu", 4096)
        opt.step()
        after = {**{n: p.detach() for n, p in model.named_parameters() if p.requires_grad},
                 **({f"film.{n}": p.detach() for n, p in film.named_parameters()} if film else {})}
        if kind == "film":
            moved = [k for k in range(K) if not torch.equal(before["film.codes"][k], after["film.codes"][k])]
            assert moved == [2], moved
            assert any(not torch.equal(before[n], after[n]) for n in before if n.endswith(".B")), "the shared LoRA moves"
        elif kind == "mcl":
            moved = {k for n in before for k in range(K) if not torch.equal(before[n][k], after[n][k])}
            assert moved == {2}, moved
        else:
            assert all(not torch.equal(before[n], after[n]) for n in before if n.endswith(".B"))
    print("routing: a rollout of mode k moves code k / adapter k only: ok")


def gen_args(**kw):
    return NS(**{"temp": 0.0, "top_p": 1.0, "top_k": 0, "max_tokens": 6, "cache": "dynamic", **kw})


def test_rollouts(d, base):
    for kind in ("film", "mcl"):
        model, film, ctx, _ = build(base, kind)
        gc = G.gen_config(gen_args(), 0, [0])
        model.generation_config = gc
        rows = [{"p": r["p"], "mode": m} for r, m in zip(rand_rows(6, [None] * 6, seed=3), [0, 1, 2, 3, 1, 0])]
        both = G.rollout(model, film, ctx, rows, gc, 0, "cpu", 6, per_mode=kind == "mcl")
        alone = [G.rollout(model, film, ctx, [r], gc, 0, "cpu", 1, per_mode=kind == "mcl")[0] for r in rows]
        assert both == alone, (kind, both, alone)
        small = 2 * (max(len(r["p"]) for r in rows) + gc.max_new_tokens)      # the token budget: calls of <= 2 rows
        assert G.rollout(model, film, ctx, rows, gc, 0, "cpu", 6, per_mode=kind == "mcl", tokens=small) == alone, kind
        same_p = [{"p": rows[0]["p"], "mode": k} for k in range(K)]
        outs = G.rollout(model, film, ctx, same_p, gc, 0, "cpu", 4, per_mode=kind == "mcl")
        assert len({tuple(t) for t, _ in outs}) > 1, f"{kind}: the modes must matter"
    model, film, ctx, _ = build(base, "none")
    p = rand_rows(1, [None])[0]["p"]
    first = G.rollout(model, film, ctx, [{"p": p, "mode": None}], G.gen_config(gen_args(), 0, [0]), 0, "cpu", 1, False)
    t0 = first[0][0]
    x = t0[1]                                           # the second greedy token as the stop token
    gc = G.gen_config(gen_args(max_tokens=5), 0, [x])
    (t, cut), = G.rollout(model, film, ctx, [{"p": p, "mode": None}], gc, 0, "cpu", 1, False)
    assert t == t0[:t0.index(x) + 1] and not cut, "cut right after the first stop token, which is kept"
    gc = G.gen_config(gen_args(max_tokens=3), 0, [299 if 299 not in first[0][0][:3] else 298])
    (t, cut), = G.rollout(model, film, ctx, [{"p": p, "mode": None}], gc, 0, "cpu", 1, False)
    assert cut and len(t) == 3, "no stop token: truncated at max_tokens"
    # T = 1 and no top-k / top-p, although the base asks for T 0.6, top-k 3, top-p 0.5
    bgc = os.path.join(d, "base_gc")
    shutil.copytree(base, bgc)
    json.dump({"do_sample": True, "temperature": 0.6, "top_k": 3, "top_p": 0.5, "eos_token_id": 0},
              open(os.path.join(bgc, "generation_config.json"), "w"))
    model, film, ctx = FM.build(bgc, 0, 4, 8, "cpu", grad_ckpt=False)
    from transformers import AutoTokenizer
    p = AutoTokenizer.from_pretrained(bgc).encode("hello world", add_special_tokens=False)
    w0 = model.model.embed_tokens.weight.detach().clone()
    for c in (2, 3, 5, 8, 13):               # logits scaled until the distribution is peaked enough to tell T apart
        with torch.no_grad():
            model.model.embed_tokens.weight.copy_(w0 * c)
            lg = model(input_ids=torch.tensor([p])).logits[0, -1].float()
        pr, p06 = torch.softmax(lg, -1), torch.softmax(lg / 0.6, -1)
        top = pr.argsort(descending=True)
        out3 = float(1 - pr[top[:3]].sum())
        if out3 > 0.1 and float(p06[top[0]] - pr[top[0]]) > 0.1:
            break
    else:
        raise AssertionError("no scale gives a usable distribution")
    gc = G.gen_config(gen_args(temp=1.0, max_tokens=1), 0, [0])
    model.generation_config = gc
    torch.manual_seed(0)
    n = 4000
    x = torch.tensor([t[0] for t, _ in G.rollout(model, film, ctx, [{"p": p, "mode": None}] * n, gc, 0, "cpu", 1000, False)])
    f1, fo = float((x == top[0]).float().mean()), float((~torch.isin(x, top[:3])).float().mean())
    se = lambda q: 4 * (q * (1 - q) / n) ** 0.5
    assert abs(f1 - float(pr[top[0]])) < se(float(pr[top[0]])), (f1, float(pr[top[0]]), float(p06[top[0]]))
    assert abs(fo - out3) < se(out3), (fo, out3)
    print("rollouts: left padding, mixed modes, stop tokens, sampling at T = 1: ok")


# ----------------------------------------------------------------------------- the trainer
def test_data(d):
    """A pool of 6 problems, M_0 of sftmulti (4 samples) and wta1 (1 per mode): p0 solved by all 8, p1 by 7."""
    data = os.path.join(d, "gdata")
    probs = problems(6)
    write_jsonl(os.path.join(data, "pool.jsonl"), probs)
    ok = lambda j, s: j == 0 or (j == 1 and s != 3)
    fake_gen(os.path.join(d, "m0_sft"), probs, 0, lambda j, m: [("so \\boxed{7}" if ok(j, s) else "\\boxed{3}", ok(j, s))
                                                                 for s in range(4)])
    fake_gen(os.path.join(d, "m0_wta"), probs, K, lambda j, m: [("so \\boxed{7}" if ok(j, 0) else "\\boxed{3}", ok(j, 0))])
    return data, [os.path.join(d, "m0_sft"), os.path.join(d, "m0_wta")]


def fake_reward(text, answer):
    return float(zlib.crc32(text.encode()) % 3 == 0)


def train_args(base, init, data, m0, out, *extra):
    return ["--base", base, "--init", init, "--data", data, "--m0", ",".join(m0), "--out", out, "--seed", "5",
            "--G", "8", "--P", "2", "--steps", "4", "--evals", "2", "--max_tokens", "6", "--lr", "1e-2", "--warmup", "1",
            "--gen_batch", "8", "--ckpt_minutes", "0", "--keep_texts", "2", *extra]


def lines(path):
    return [json.loads(x) for x in open(path)]


def test_trainer(d, base):
    from tests.test_phase1_self import adapter
    data, m0 = test_data(d)
    pool, dropped = G.train_pool(data, m0, 1)
    assert [p["id"] for p in pool] == ["p1", "p2", "p3", "p4", "p5"] and dropped == 1, "p0: every M_0 attempt right"
    assert len(G.train_pool(data, m0, 0)[0]) == 6
    inits = {"none": adapter(base, os.path.join(d, "g_none.pt")), "film": adapter(base, os.path.join(d, "g_film.pt"), film_modes=K),
             "mcl": adapter(base, os.path.join(d, "g_mcl.pt"), lora_modes=K)}
    orig = G.reward
    G.reward = fake_reward
    try:
        runs = {}
        for kind, init in inits.items():
            out = os.path.join(d, f"gt_{kind}")
            assert G.main(train_args(base, init, data, m0, out)) == "done"
            runs[kind] = out
        steps = {k: lines(f"{o}/steps.jsonl") for k, o in runs.items()}
        rolls = {k: lines(f"{o}/rollouts.jsonl") for k, o in runs.items()}
        for kind in runs:
            assert len(rolls[kind]) == 4 * 2 * 8 and [s["step"] for s in steps[kind]] == [0, 1, 2, 3]
            res = json.load(open(f"{runs[kind]}/result.json"))
            assert res["rollouts"] == 64 and res["complete"] and res["ckpts"] == [2, 4] and res["dropped_easy"] == 1
            assert res["gen_tokens"] == sum(r["len"] for r in rolls[kind]) == sum(s["gen_tokens"] for s in steps[kind])
            for t in range(4):
                ids = [r["id"] for r in rolls[kind] if r["step"] == t]
                assert ids == [p["id"] for i in G.step_problems(5, 5, 2, t) for p in [pool[i]] for _ in range(8)]
            assert [r["id"] for r in rolls[kind]] == [r["id"] for r in rolls["none"]], "the same problems for every chain"
            for t, s in enumerate(steps[kind]):
                rs = np.array([r["r"] for r in rolls[kind] if r["step"] == t]).reshape(2, 8)
                assert s["trained"] == sum(8 for g in rs if g.std() > 0) and abs(s["reward"] - rs.mean()) < 1e-12
                assert s["mixed"] == np.mean(rs.std(1) > 0)
            if kind != "none":
                assert all([r["mode"] for r in rolls[kind]][i:i + 8] == [0, 0, 1, 1, 2, 2, 3, 3] for i in range(0, 64, 8))
                assert len(steps[kind][0]["reward_mode"]) == K
            else:
                assert all(r["mode"] is None for r in rolls[kind])
            for j, step in ((1, 2), (2, 4)):
                ck = torch.load(f"{runs[kind]}/ckpt{j}/adapter.pt", weights_only=False)
                ref = torch.load(inits[kind], weights_only=False)
                assert ck["step"] == step and set(ck["adapter"]) == set(ref["adapter"])
                assert ck["cfg"]["film_modes"] == ref["cfg"]["film_modes"] and ck["cfg"]["lora_modes"] == ref["cfg"]["lora_modes"]
                assert ck["cfg"]["prefix_modes"] == 0 and ck["cfg"]["lora_r"] == 4 and ck["cfg"]["lora_alpha"] == 8
            assert sorted(os.listdir(f"{runs[kind]}/texts")) == ["step00000.jsonl.gz", "step00002.jsonl.gz", "step00003.jsonl.gz"]
            assert not os.path.exists(f"{runs[kind]}/last.pt") and os.path.exists(f"{runs[kind]}/DONE")
        assert any(s["trained"] for s in steps["film"]), "the fake reward must give mixed groups"
        # a run stopped after 1 and after 3 steps and resumed equals the uninterrupted one
        for kind in ("film", "mcl"):
            out = os.path.join(d, f"gt_{kind}_resumed")
            assert G.main(train_args(base, inits[kind], data, m0, out, "--stop_after", "1")) == "stopped"
            assert not os.path.exists(f"{out}/DONE") and os.path.exists(f"{out}/last.pt")
            with open(f"{out}/rollouts.jsonl", "a") as f:            # a step logged but not checkpointed when killed
                f.write(json.dumps({"step": 1, "id": "junk"}) + "\n" + '{"step": 1, "id"')
            assert G.main(train_args(base, inits[kind], data, m0, out, "--stop_after", "2")) == "stopped"
            assert G.main(train_args(base, inits[kind], data, m0, out)) == "done"
            assert lines(f"{out}/rollouts.jsonl") == rolls[kind] and \
                [{k: v for k, v in s.items() if not k.endswith("_s")} for s in lines(f"{out}/steps.jsonl")] == \
                [{k: v for k, v in s.items() if not k.endswith("_s")} for s in steps[kind]]
            a, b = (torch.load(f"{o}/ckpt2/adapter.pt", weights_only=False)["adapter"] for o in (out, runs[kind]))
            assert all(torch.equal(a[n], b[n]) for n in a), f"{kind}: resumed != uninterrupted"
        # the safety time limit: the last adapter becomes the final checkpoint, last.pt stays for a resume
        out = os.path.join(d, "gt_timeout")
        assert G.main(train_args(base, inits["film"], data, m0, out, "--stop_after", "1")) == "stopped"
        assert G.main(train_args(base, inits["film"], data, m0, out, "--max_hours", "1e-7")) == "done"
        res = json.load(open(f"{out}/result.json"))
        assert res["stopped"] == "time" and not res["complete"] and res["steps_done"] == 1
        assert os.path.exists(f"{out}/ckpt2/adapter.pt") and os.path.exists(f"{out}/last.pt") and os.path.exists(f"{out}/DONE")
        # sampling other than pi itself is refused
        for bad in (("--temp", "0.7"), ("--top_p", "0.9"), ("--top_k", "20")):
            try:
                G.main(train_args(base, inits["none"], data, m0, os.path.join(d, "gt_temp"), *bad))
                raise AssertionError(f"{bad} must be refused")
            except SystemExit as e:
                assert "--temp 1" in str(e), e
        # a different setting in the same directory is refused
        out = os.path.join(d, "gt_film_resumed")
        os.remove(f"{out}/DONE")
        try:
            G.main(train_args(base, inits["film"], data, m0, out, "--lr", "2e-2"))
            raise AssertionError("a changed LR must be refused")
        except SystemExit as e:
            assert "lr" in str(e)
        # equal rewards everywhere: no row trained, the adapter does not move
        G.reward = lambda text, answer: 1.0
        out = os.path.join(d, "gt_equal")
        G.main(train_args(base, inits["film"], data, m0, out))
        assert all(s["trained"] == 0 and s["grad_norm"] == 0 and s["mixed"] == 0 and s["all1"] == 1 for s in lines(f"{out}/steps.jsonl"))
        a, b = torch.load(f"{out}/ckpt2/adapter.pt", weights_only=False)["adapter"], torch.load(inits["film"], weights_only=False)["adapter"]
        assert all(torch.equal(a[n], b[n].float()) for n in a)
        # the cover ablation trains, modes required
        G.reward = fake_reward
        assert G.main(train_args(base, inits["film"], data, m0, os.path.join(d, "gt_cover"), "--adv", "cover")) == "done"
        try:
            G.main(train_args(base, inits["none"], data, m0, os.path.join(d, "gt_cover_none"), "--adv", "cover"))
            raise AssertionError("cover without modes must be refused")
        except SystemExit:
            pass
        # calibration -> pinned steps: the slowest chain fits the hours, the first writer wins
        cal = []
        for kind in ("none", "film"):
            out = os.path.join(d, f"gcal_{kind}")
            G.main(train_args(base, inits[kind], data, m0, out, "--calib", "2"))
            c = json.load(open(f"{out}/calib.json"))
            assert c["steps"] == 2 and c["min_per_step"] > 0 and len(c["step_s"]) == 2 and not os.path.exists(f"{out}/ckpt1")
            assert len(lines(f"{out}/rollouts.jsonl")) == 2 * 2 * 8
            cal.append(f"{out}/calib.json")
        sf = os.path.join(d, "gsteps.json")
        mps = max(json.load(open(f))["min_per_step"] for f in cal)
        hours = 6 * 1.1 * mps / 60                  # 6 steps of the slowest: S = 6 (a multiple of the 2 checkpoints)
        auto = ["--steps", "0", "--steps_file", sf, "--calib_from", ",".join(cal), "--train_hours", f"{hours * 1.0001}",
                "--steps_max", "50"]
        assert G.pinned_steps(G.parse(train_args(base, inits["none"], data, m0, "x") + auto)) == 6
        json.dump({"steps": 4}, open(sf, "w"))
        assert G.main(train_args(base, inits["none"], data, m0, os.path.join(d, "gt_pinned"), *auto)) == "done"
        assert json.load(open(os.path.join(d, "gt_pinned", "result.json")))["steps"] == 4, "steps.json wins"
    finally:
        G.reward = orig
    print("trainer: accounting, equal problems, resume, refused settings, calibration, filter: ok")


# ----------------------------------------------------------------------------- the run
def fake_self(d, base, p1, seeds_run, sizes=None, m0=True):
    """phase1_self with data (and the M_0 generation directories of seeds 0 and 1)."""
    sr = os.path.join(d, "self")
    n = sizes or {"pool": 12, "hv": 6, "test": 5}
    rows = {k: [dict(p, stratum=["A", "B0", "B12", "B36"][i % (4 if k == "pool" else 3)])
                for i, p in enumerate(problems(v, prefix=k[0]))] for k, v in n.items()}
    for k, v in rows.items():
        write_jsonl(f"{sr}/data/{k}.jsonl", v)
    json.dump({"sizes": n}, open(f"{sr}/data/meta.json", "w"))
    if m0:
        a = NS(phase1=p1, seeds_run=seeds_run)
        for s in (0, 1):
            for st in ("sftmulti", "wta1"):
                g = f"{sr}/s{s}/start_{st}/gen"
                os.makedirs(g, exist_ok=True)
                ad = os.path.abspath(srun_start(a, s, st))
                json.dump({"modes": 0 if st == "sftmulti" else K, "sets": grun.M0_SETS, "seed": 0, "adapter": ad},
                          open(f"{g}/gen.json", "w"))
                json.dump({"adapter": ad, "adapter_bytes": os.path.getsize(ad), "adapter_mtime": int(os.path.getmtime(ad)),
                           "base": os.path.abspath(base), "data": os.path.abspath(f"{sr}/data"), "seed": 0,
                           "max_tokens": 1200}, open(f"{g}/keval.json", "w"))
                open(f"{g}/DONE", "w").close()
    return sr


def srun_start(a, s, st):
    from phase1.srun import start_adapter
    return start_adapter(a, s, st)


def grun_cmd(run, sr, p1, seeds_run, base, *extra):
    return [sys.executable, "-m", "phase1.grun", "--run", run, "--self_run", sr, "--phase1", p1, "--seeds_run",
            seeds_run, "--model", base, *extra]


def test_plan(d, base):
    p1, seeds_run = os.path.join(d, "p1_plan"), os.path.join(d, "seeds_plan")
    fake_refs(p1, seeds_run, base, real=False)
    for v in ("wta1", "sftmulti", "loramcl"):
        os.makedirs(f"{seeds_run}/train/{v}_s1", exist_ok=True)
        open(f"{seeds_run}/train/{v}_s1/adapter.pt", "w").close()
    sr = fake_self(os.path.join(d, "plan_self"), base, p1, seeds_run, {"pool": 2400, "hv": 800, "test": 1500})
    run = os.path.join(d, "gplan")
    out = subprocess.run(grun_cmd(run, sr, p1, seeds_run, base, "--plan_only"), env=ENV, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-3000:]
    out = out.stdout
    print(out.splitlines()[0])
    # 2 calibrations; per seed and chain a training and 3 evaluations; the report
    assert f"{2 + 2 * 2 * 4 + 1} jobs" in out, out
    hours = float(out.split("simulated ")[1].split(" h")[0])
    assert 10 <= hours <= 12.5, hours
    S = int(out.split("; ")[2].split(" steps")[0])
    assert 280 <= S <= 330, S
    line = lambda name: next(x for x in out.splitlines() if f"  {name}:" in x)
    assert "deps ['calib_sftmulti', 'calib_wta1']" in line("train_s1_grpo_wta1") and "deps []" in line("calib_wta1")
    assert "deps ['train_s0_grpo']" in line("gen_s0_grpo_c3") and "prio 30" in line("gen_s0_grpo_c3")
    assert "prio 20" in line("gen_s0_grpo_c1") and "prio 29" in line("gen_s1_grpo_c3")
    a = grun.parse(["--self_run", sr, "--phase1", p1, "--seeds_run", seeds_run])
    J = {j.name: j.cmd for j in grun.jobs(a, run, base)}
    c = J["train_s1_grpo_wta1"]
    arg = lambda c, k: c[c.index(k) + 1]
    assert arg(c, "--init") == f"{seeds_run}/train/wta1_s1/adapter.pt" and arg(c, "--seed") == "1"
    assert arg(c, "--m0").split(",") == [f"{sr}/s1/start_sftmulti/gen", f"{sr}/s1/start_wta1/gen"]
    assert arg(c, "--steps_file") == f"{run}/steps.json" and arg(c, "--adv") == "mean"
    assert arg(c, "--calib_from").split(",") == [f"{run}/calib/sftmulti/calib.json", f"{run}/calib/wta1/calib.json"]
    c0 = J["train_s1_grpo"]
    same = lambda k: arg(c, k) == arg(c0, k)
    assert all(same(k) for k in ("--data", "--m0", "--seed", "--G", "--P", "--lr", "--temp", "--max_tokens", "--steps_file"))
    c = J["gen_s0_grpo_wta1_c3"]
    assert arg(c, "--sets") == "hv:16,test:4" and arg(c, "--drop_adapter") == "0" and arg(c, "--data") == f"{sr}/data"
    assert arg(c, "--adapter") == f"{run}/s0/grpo_wta1/train/ckpt3/adapter.pt"
    c = J["gen_s0_grpo_wta1_c1"]
    assert arg(c, "--sets") == "hv:16" and arg(c, "--drop_adapter") == "1"
    seeds = {arg(c, "--seed") for n, c in J.items() if n.startswith("gen_")}
    assert len(seeds) == 2 * 2 * 3, "every evaluation samples with its own seed"
    J = {j.name: j.cmd for j in grun.jobs(grun.parse(["--self_run", sr, "--phase1", p1, "--seeds_run", seeds_run,
                                                      "--steps", "300", "--chains", "grpo,grpo_wta1,grpo_wta1c"]), run, base)}
    assert not any(n.startswith("calib") for n in J) and arg(J["train_s0_grpo_wta1c"], "--steps") == "300"
    assert arg(J["train_s0_grpo_wta1c"], "--adv") == "cover"
    st = json.load(open(f"{run}/state.json")) if os.path.exists(f"{run}/state.json") else {}
    json.dump({**st, "G": 8}, open(f"{run}/state.json", "w"))
    r = subprocess.run(grun_cmd(run, sr, p1, seeds_run, base, "--plan_only"), env=ENV, capture_output=True, text=True)
    assert r.returncode != 0 and "G = 8" in r.stderr, r.stderr
    os.remove(f"{sr}/s1/start_wta1/gen/DONE")
    r = subprocess.run(grun_cmd(os.path.join(d, "gplan2"), sr, p1, seeds_run, base, "--plan_only"), env=ENV,
                       capture_output=True, text=True)
    assert r.returncode != 0 and "s1/start_wta1/gen (M_0" in r.stderr, r.stderr
    open(f"{sr}/s1/start_wta1/gen/DONE", "w").close()
    json.dump({"modes": K, "sets": grun.M0_SETS, "adapter": os.path.abspath(f"{seeds_run}/train/sftmulti_s1/adapter.pt")},
              open(f"{sr}/s1/start_wta1/gen/gen.json", "w"))
    r = subprocess.run(grun_cmd(os.path.join(d, "gplan3"), sr, p1, seeds_run, base, "--plan_only"), env=ENV,
                       capture_output=True, text=True)
    assert r.returncode != 0 and "not by the start adapter" in r.stderr, r.stderr
    # M_0 generated with another max_tokens or by another version of the start adapter is refused
    ad = os.path.abspath(f"{seeds_run}/train/wta1_s1/adapter.pt")
    json.dump({"modes": K, "sets": grun.M0_SETS, "adapter": ad}, open(f"{sr}/s1/start_wta1/gen/gen.json", "w"))
    kf = f"{sr}/s1/start_wta1/gen/keval.json"
    kv = json.load(open(kf))
    json.dump({**kv, "max_tokens": 1000, "adapter_bytes": kv["adapter_bytes"] + 1}, open(kf, "w"))
    r = subprocess.run(grun_cmd(os.path.join(d, "gplan4"), sr, p1, seeds_run, base, "--plan_only"), env=ENV,
                       capture_output=True, text=True)
    assert r.returncode != 0 and "max_tokens 1000" in r.stderr and "another version" in r.stderr, r.stderr
    json.dump(kv, open(kf, "w"))
    r = subprocess.run(grun_cmd(os.path.join(d, "gplan5"), sr, p1, seeds_run, base, "--plan_only", "--temp", "0.7"),
                       env=ENV, capture_output=True, text=True)
    assert r.returncode != 0 and "--temp 1" in r.stderr, r.stderr
    print("plan: ok")


def test_smoke(d, base):
    p1, seeds_run = os.path.join(d, "p1s"), os.path.join(d, "seeds_s")
    fake_refs(p1, seeds_run, base, real=True)
    sr = fake_self(os.path.join(d, "smoke_self"), base, p1, seeds_run, m0=False)
    run = os.path.join(d, "gsmoke")
    chains = list(grun.CHAINS)                 # every chain: the export of separate adapters through selftrain gen too
    r = subprocess.run(grun_cmd(run, sr, p1, seeds_run, base, "--smoke", "--max_tokens", "8", "--gpus", "0,1,2,3",
                                "--chains", ",".join(chains)), env=ENV)
    assert r.returncode == 0, "smoke run failed, see " + os.path.join(run, "logs")
    assert json.load(open(f"{run}/steps.json"))["steps"] == 2
    for st in ("sftmulti", "wta1", "loramcl"):
        assert json.load(open(f"{run}/calib/{st}/calib.json"))["steps"] == 2
        g = json.load(open(f"{run}/m0/s0/start_{st}/gen/gen.json"))
        assert g["sets"] == grun.M0_SETS and g["modes"] == (0 if st == "sftmulti" else K)
    rolls = {}
    for c in chains:
        t = f"{run}/s0/{c}/train"
        res = json.load(open(f"{t}/result.json"))
        assert res["steps_done"] == res["steps"] == 2 and res["rollouts"] == 2 * 2 * 4 and res["ckpts"] == [1, 2]
        assert not os.path.exists(f"{t}/ckpt1/adapter.pt"), "intermediate adapters are dropped after their evaluation"
        assert os.path.exists(f"{t}/ckpt2/adapter.pt") and not os.path.exists(f"{t}/last.pt")
        rolls[c] = lines(f"{t}/rollouts.jsonl")
        assert len(rolls[c]) == 16
        for j, sets in ((1, {"hv": 16}), (2, {"hv": 16, "test": 4})):
            g = json.load(open(f"{run}/s0/{c}/c{j}/gen/gen.json"))
            assert g["sets"] == sets and g["modes"] == (0 if c == "grpo" else K)
    assert all([x["id"] for x in rolls["grpo"]] == [x["id"] for x in rolls[c]] for c in chains)
    assert all([x["mode"] for x in rolls[c]][:4] == [0, 1, 2, 3] for c in chains if c != "grpo")
    text = open(f"{run}/report_grpo.md", encoding="utf-8").read()
    print("\n".join(text.splitlines()[:4]))
    for row in ("## 1. Главная метрика", "| grpo_wta1 − grpo, уровень (главное) |", "| grpo_wta1 − grpo, изменение от M_0 "
                "(главное) |", "| grpo | Δ |", "| s0 grpo_wta1 шаг 2 |", "## 7. Обрезанные попытки", "| s0 grpo | 2 |",
                "Раздел пропущен: нет phase1_self", "| grpo_wta1c − grpo_wta1, изменение от M_0 |",
                "| grpo_wta1 − grpo_loramcl, уровень |", "| s0 grpo_loramcl шаг 1 |", "- шаги по калибровке: 2 (",
                "| s0 grpo (псевдорежимы) M_0 |", "| s0 grpo (псевдорежимы) шаг 2 |",
                "(предварительно: нет seed [1])"):         # planned seeds 0 and 1, the smoke run has seed 0 only
        assert row in text, row
    assert "нет финального" not in text and "не выполнены" not in text and "отклонены" not in text, text
    summ = json.load(open(f"{run}/summary_grpo.json"))
    assert summ["complete_seeds"] == [0] and "primary:grpo_wta1-grpo:level" in summ["comparisons"]
    assert set(summ["comparisons"]["passk:grpo"]) == {"M0", "final", "delta", "first_k_not_better"}
    # the comparison with RFT: the final models of the chains of phase1_self on the same hv
    hv = [json.loads(x) for x in open(f"{run}/data/hv.jsonl")]
    json.dump({"rounds": 3}, open(f"{sr}/state.json", "w"))
    for c, M in (("rftbal", 0), ("rft", 0), ("wta1", K)):
        fake_gen(f"{sr}/s0/{c}/r3/gen", hv, M, lambda j, m: [("\\boxed{7}", j % 2 == 0)] * (4 if M else 16), name="hv")
    greport.build(run, f"{run}/data", f"{run}/m0", sr, [0, 1], ["grpo", "grpo_wta1"], 2, boot_n=20)
    text = open(f"{run}/report_grpo.md", encoding="utf-8").read()
    rows = [x for x in text.splitlines() if x.startswith("| grpo − rftbal |") or x.startswith("| grpo_wta1 − wta1 |")]
    assert len(rows) == 2 and all("—" not in x.split("|")[2] for x in rows), rows
    assert "предварительно: нет seed [1]" in text and "отчёта phase1_self нет" in text
    print("smoke run end to end, report, comparison with RFT: ok")


def main():
    torch.manual_seed(0)
    d = tempfile.mkdtemp()
    base = tiny_model(d)
    test_math()
    test_loss(base)
    test_routing(base)
    test_rollouts(d, base)
    test_trainer(d, base)
    test_plan(d, base)
    test_smoke(d, base)
    print("all ok")


if __name__ == "__main__":
    main()
