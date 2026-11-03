"""Equal-parameter run on CPU with a tiny random Qwen3 and fabricated r = 32 results (~3-5 min).

    PYTHONUTF8=1 python3 -m tests.test_phase1_rank

1. alpha by rank, trainable parameters on the Qwen3-1.7B shapes, export with an alpha other than 2r
   (LoRA + FiLM and K adapters, several modes)
2. --init and last.pt keep the alpha of the adapter, a resumed run keeps the RNG and the mode counts
3. plan: jobs, commands, simulated time, missing r = 32 evaluations, a changed alpha rule
4. the --smoke run end to end through the real job runner (train, evaluate on the HF backend, report)"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

from phase1 import film as FM
from phase1 import rank_run
from tests.test_phase1_seeds import ENV, K, fake_phases, problems, tiny_model, write_modes, write_single


def test_alpha_params_export(base, d):
    assert rank_run.alpha_of(32, "rs") == 64 and rank_run.alpha_of(128, "rs") == 128
    assert abs(rank_run.alpha_of(10, "rs") - 35.777) < 1e-3 and rank_run.alpha_of(10, "2r") == 20
    from transformers import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config(vocab_size=151936, hidden_size=2048, intermediate_size=6144, num_hidden_layers=28,
                      num_attention_heads=16, num_key_value_heads=8, head_dim=128, tie_word_embeddings=True)
    with torch.device("meta"):
        m = Qwen3ForCausalLM(cfg)
        per_r = sum(lin.in_features + lin.out_features for layer in m.model.layers
                    for parent in (layer.self_attn, layer.mlp) for n in FM.LORA_TARGETS
                    if (lin := getattr(parent, n, None)) is not None)
        film = sum(p.numel() for p in FM.FiLM(len(FM.norm_modules(m)), 2048, K=4).parameters())
    M = {"wta1_r32": (32 * per_r + film) / 1e6, "loramcl_r32": 4 * 32 * per_r / 1e6,
         "wta1_r128": (128 * per_r + film) / 1e6, "loramcl_r10": 4 * 10 * per_r / 1e6}
    print({k: round(v, 1) for k, v in M.items()})
    assert round(M["wta1_r32"], 1) == 42.5 and round(M["loramcl_r32"], 1) == 139.5, "as in report_seeds"
    assert abs(M["wta1_r128"] / M["loramcl_r32"] - 1) < 0.06 and abs(M["loramcl_r10"] / M["wta1_r32"] - 1) < 0.03
    # export merges with the scale of the adapter, not with 2: LoRA + FiLM (wta1), K adapters (LoRA-MCL), several
    # modes exported one after another, each against the training model
    from transformers import AutoModelForCausalLM
    x = torch.randint(0, 300, (2, 12))
    r, alpha = 4, rank_run.alpha_of(4, "rs")
    for film_modes, lora_modes, modes in ((K, 0, [0, 2]), (0, K, [1, 3])):
        model, film, ctx = FM.build(base, film_modes, r, alpha, "cpu", grad_ckpt=False, lora_modes=lora_modes)
        with torch.no_grad():
            for n, p in list(model.named_parameters()) + ([("film." + n, p) for n, p in film.named_parameters()]
                                                          if film else []):
                if n.endswith(".B") or n == "film.net.2.weight":
                    p.normal_(0, 0.1)
        path = os.path.join(d, f"alpha{lora_modes}.pt")
        torch.save({"adapter": FM.adapter_state(model, film), "cfg": {"film_modes": film_modes, "lora_r": r,
                                                                     "lora_alpha": alpha, "lora_modes": lora_modes}}, path)
        outs = FM.export_modes(base, path, [os.path.join(d, f"alpha{lora_modes}_{m}") for m in modes], modes,
                               dtype=torch.float32)
        model.eval()
        got = {}
        for m, out in zip(modes, outs):
            with torch.no_grad():
                md = torch.full((2,), m)
                ctx.modes = md if lora_modes else None
                ctx.gamma = film.gamma(md) if film else None
                want = model(input_ids=x).logits
                got[m] = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32)(input_ids=x).logits
            assert float((want - got[m]).abs().max()) < 1e-4, (film_modes, lora_modes, m)
        assert float((got[modes[0]] - got[modes[1]]).abs().max()) > 1e-3, "exported modes must differ"
    print("alpha by rank, parameters, export with alpha: ok")


def test_init_and_resume(d, p1, base):
    """--init and last.pt keep the alpha of the adapter; a resumed run keeps the WTA RNG and the mode counts."""
    def train(out, *extra, check=True):
        return subprocess.run([sys.executable, "-m", "phase1.train", "--variant", "wta1", "--base", base, "--data",
                               f"{p1}/data", "--out", out, "--P", "2", "--max_steps", "2", "--lora_r", "4", *extra],
                              env=ENV, capture_output=True, text=True, check=check)
    first = os.path.join(d, "first")
    train(first, "--lora_alpha", "22.6")
    ck = torch.load(f"{first}/adapter.pt", map_location="cpu", weights_only=False)
    assert ck["cfg"]["lora_alpha"] == 22.6 and "opt" in ck
    cont = os.path.join(d, "cont")
    train(cont, "--init", f"{first}/adapter.pt")
    assert json.load(open(f"{cont}/result.json"))["lora_alpha"] == 22.6, "alpha of --init"
    r = train(os.path.join(d, "cont_bad"), "--init", f"{first}/adapter.pt", "--lora_alpha", "8", check=False)
    assert r.returncode != 0 and "alpha 22.6" in r.stderr, r.stderr
    # a stopped run: last.pt after step 1 with counts of mode 0 and a known RNG state
    import random
    rng = random.Random(5)
    rng.random()
    for alpha, ok in (("8", False), ("22.6", True)):
        out = os.path.join(d, f"resume_{alpha}")
        os.makedirs(out)
        torch.save({"adapter": ck["adapter"], "opt": ck["opt"], "state": {"step": 1, "tokens": 10, "elapsed": 0.0},
                    "cfg": ck["cfg"], "rng": rng.getstate(), "share": torch.tensor([1000.0, 0, 0, 0])},
                   f"{out}/last.pt")
        r = train(out, "--lora_alpha", alpha, check=False)
        assert (r.returncode == 0) == ok, r.stderr
    res = json.load(open(f"{out}/result.json"))
    assert res["steps_done"] == 2 and res["mode_share"][0] > 0.9, res
    assert "resumed at step 1" in r.stdout + r.stderr
    print("alpha of --init and last.pt, resumed counts: ok")


def fake_refs(d, p1, seeds_run, seeds, n):
    """r = 32 evaluations and trainings of every seed (seed 0 of wta1 and sftmulti in phase 1)."""
    test = problems(n)
    rng = np.random.default_rng(1)
    for s in seeds:
        for v in ("wta1", "loramcl", "sftmulti"):
            old = s == 0 and v != "loramcl"
            e = f"{p1}/eval/{v}" if old else f"{seeds_run}/eval/{v}_s{s}"
            t = f"{p1}/train/{v}" if old else f"{seeds_run}/train/{v}_s{s}"
            if v == "sftmulti":
                write_single(e, test, rng.random((8, n)) < 0.5, rng.integers(0, 3, (8, n)))
            else:
                write_modes(e, test, rng.random((K, 2, n)) < 0.5, rng.integers(0, 3, (K, 2, n)))
            os.makedirs(t, exist_ok=True)
            json.dump({"steps_done": 400, "epochs_done": 0.5, "tokens": 6.6e6, "hours": 2.0, "trainable_M": 139.5,
                       "mode_share": [0.25] * K}, open(f"{t}/result.json", "w"))


def rank_cmd(run, p0, p1, seeds_run, base, *extra):
    return [sys.executable, "-m", "phase1.rank_run", "--run", run, "--phase0", p0, "--phase1", p1,
            "--seeds_run", seeds_run, "--model", base, *extra]


def test_plan(d, p0, p1, base):
    seeds_run = os.path.join(d, "seeds_run_plan")
    run = os.path.join(d, "plan")
    r = subprocess.run(rank_cmd(run, p0, p1, seeds_run, base, "--plan_only"), env=ENV, capture_output=True, text=True)
    assert r.returncode != 0 and "loramcl_s0" in r.stderr, r.stderr
    fake_refs(d, os.path.join(d, "p1_plan"), seeds_run, (0, 1), 30)
    p1b = os.path.join(d, "p1_plan")
    os.makedirs(f"{p1b}/data", exist_ok=True)
    for f in ("train", "verify", "test"):
        open(f"{p1b}/data/{f}.jsonl", "w").write(open(f"{p1}/data/{f}.jsonl").read())
    out = subprocess.run(rank_cmd(run, p0, p1b, seeds_run, base, "--plan_only"), env=ENV, capture_output=True,
                         text=True, check=True).stdout
    print(out.splitlines()[0])
    assert "9 jobs" in out and "'loramcl_r10': 35.78" in out, out
    hours = float(out.split("simulated ")[1].split(" h")[0])
    assert 3.0 <= hours <= 3.5, hours
    assert "eval_wta1_r128_s1: est 1.00 h, deps ['train_wta1_r128_s1']" in out
    a = argparse.Namespace(smoke=False, seeds=[0, 1], alpha_rule="rs", max_tokens=1200, seeds_run=seeds_run,
                           phase1=p1b, phase0=p0)
    J = {j.name: j.cmd for j in rank_run.jobs(a, run, f"{p1b}/data", base)}
    c = J["train_wta1_r128_s1"]
    assert c[c.index("--lora_r") + 1] == "128" and c[c.index("--lora_alpha") + 1] == "128.0000"
    assert c[c.index("--variant") + 1] == "wta1" and c[c.index("--seed") + 1] == "1"
    c = J["train_loramcl_r10_s0"]
    assert c[c.index("--lora_r") + 1] == "10" and c[c.index("--lora_alpha") + 1] == "35.7771"
    c = J["eval_loramcl_r10_s1"]
    assert c[c.index("--seed") + 1] == "10017", "the sampling seed of phase1_seeds"
    os.makedirs(run, exist_ok=True)
    json.dump({"alpha_rule": "2r"}, open(f"{run}/state.json", "w"))
    r = subprocess.run(rank_cmd(run, p0, p1b, seeds_run, base, "--plan_only"), env=ENV, capture_output=True, text=True)
    assert r.returncode != 0 and "alpha_rule = 2r" in r.stderr, r.stderr
    print("plan: ok")


def test_smoke(d, p0, p1, base):
    seeds_run = os.path.join(d, "seeds_run")
    fake_refs(d, p1, seeds_run, (1,), 16)              # the smoke run uses seed 1 and the first 16 test problems
    run = os.path.join(d, "smoke")
    r = subprocess.run(rank_cmd(run, p0, p1, seeds_run, base, "--smoke", "--max_tokens", "8", "--gpus", "0,1"), env=ENV)
    assert r.returncode == 0, "smoke run failed, see " + os.path.join(run, "logs")
    for name, r_, alpha in (("wta1_r128", 128, 128.0), ("loramcl_r10", 10, 35.7771)):
        res = json.load(open(f"{run}/train/{name}_s1/result.json"))
        assert res["lora_r"] == r_ and abs(res["lora_alpha"] - alpha) < 1e-3 and res["steps_done"] == 2, res
        ck = torch.load(f"{run}/train/{name}_s1/adapter.pt", map_location="cpu", weights_only=False)
        assert ck["cfg"]["lora_r"] == r_ and abs(ck["cfg"]["lora_alpha"] - alpha) < 1e-3 and "opt" not in ck
        a = next(v for n, v in ck["adapter"].items() if n.endswith(".A"))
        assert a.shape[-2] == r_
        assert os.path.exists(f"{run}/eval/{name}_s1/DONE") and not os.path.exists(f"{run}/eval/{name}_s1/tmp_export")
    assert ck["cfg"]["lora_modes"] == K
    text = open(f"{run}/report_rank.md", encoding="utf-8").read()
    print("\n".join(text.splitlines()[:16]))
    for row in ("| wta1_r128 (", "| loramcl_r10 (", "| wta1 (", "| loramcl (", "| sftmulti (",
                "| wta1_r128 − loramcl |", "| wta1 − loramcl_r10 |", "| wta1 − loramcl |", "| wta1_r128 | 1 | 128 | 128 |"):
        assert row in text, row
    assert "нет данных" not in text, text
    summ = json.load(open(f"{run}/summary_rank.json"))
    for X, R in (("wta1_r128", "loramcl"), ("wta1", "loramcl_r10"), ("loramcl_r10", "loramcl")):
        for k in ("pass1", "pass4", "same_wrong"):
            c = summ["comparisons"][f"{X}-{R}:{k}"]
            want = summ["summary"][X][k]["per_seed"]["1"] - summ["summary"][R][k]["per_seed"]["1"]
            assert c["n"] == 1 and abs(c["mean"] - want) < 1e-12 and c["lo"] <= c["hi"], (X, R, k, c)
    print("smoke run end to end: ok")


def main():
    d = tempfile.mkdtemp()
    base = tiny_model(d)
    test_alpha_params_export(base, d)
    p0, p1 = fake_phases(d)
    test_init_and_resume(d, p1, base)
    test_plan(d, p0, p1, base)
    test_smoke(d, p0, p1, base)
    print("all ok")


if __name__ == "__main__":
    main()
