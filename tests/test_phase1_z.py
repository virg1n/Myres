"""Continuous-z run on CPU with a tiny random Qwen3 and tiny reference adapters (~3-5 min).

    PYTHONUTF8=1 python3 -m tests.test_phase1_z

1. scale of z and of the first FiLM layer at every zdim, trainable parameters on the Qwen3-1.7B shapes
2. a checkpoint of one z (shared export + norm shard) reproduces the training model; the same for a FiLM code
3. wtaz training: assignment, outputs, reproducible z; phase1.evaluate refuses a continuous-z adapter
4. pass@k / maj@k estimators against brute force
5. plan: jobs, commands, simulated time, missing references, a changed k
6. the --smoke run end to end through the real job runner (train, keval on the HF backend, report)"""
import itertools
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

from phase1 import film as FM
from phase1 import zreport
from phase1.train import assign
from tests.test_phase1_seeds import ENV, K, fake_phases, tiny_model


def test_scale_params():
    torch.manual_seed(0)
    ref = FM.FiLM(3, 8, K=20000)                          # many codes: the spread of W c of wta1 at the start
    want = float((ref.codes @ ref.net[0].weight.t()).detach().std())
    for zd in (8, 16, 128, 1024):
        f = FM.FiLM(3, 8, zdim=zd, continuous=True)
        z = f.sample_z(20000, torch.Generator().manual_seed(1))
        assert abs(float(z.norm(dim=1).mean()) - 1) < 0.05, zd
        got = float((f.zscale * z @ f.net[0].weight.t()).detach().std())
        assert abs(got / want - 1) < 0.08, (zd, got, want)
        assert f.codes is None and "codes" not in dict(f.named_parameters())
        # the first layer of wta1 whatever zdim (bound 1/4); at zdim 16 the input zscale * z is distributed as a code
        assert float(f.net[0].weight.abs().max()) <= 0.25 and float(f.net[0].weight.abs().max()) > 0.24
        assert float(f.net[0].bias.abs().max()) <= 0.25
    f = FM.FiLM(3, 8, zdim=16, continuous=True)
    assert abs(float((f.zscale * f.sample_z(20000, torch.Generator().manual_seed(2))).std()) - 1) < 0.02
    from transformers import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config(vocab_size=151936, hidden_size=2048, intermediate_size=6144, num_hidden_layers=28,
                      num_attention_heads=16, num_key_value_heads=8, head_dim=128, tie_word_embeddings=True)
    with torch.device("meta"):
        m = Qwen3ForCausalLM(cfg)
        lora = 32 * sum(lin.in_features + lin.out_features for layer in m.model.layers
                        for parent in (layer.self_attn, layer.mlp) for n in FM.LORA_TARGETS
                        if (lin := getattr(parent, n, None)) is not None)
        n = len(FM.norm_modules(m))
        wta1 = lora + sum(p.numel() for p in FM.FiLM(n, 2048, K=4).parameters())
        z = {zd: lora + sum(p.numel() for p in FM.FiLM(n, 2048, zdim=zd, continuous=True).parameters())
             for zd in (8, 16, 128, 1024)}
    print({"wta1": round(wta1 / 1e6, 2), **{f"wtaz_d{k}": round(v / 1e6, 2) for k, v in z.items()}})
    assert round(wta1 / 1e6, 1) == 42.5 and wta1 - z[16] == 64, "only the 4 x 16 codes are gone"
    assert z[1024] - z[16] == 64 * (1024 - 16) and abs(z[1024] / wta1 - 1) < 0.002
    print("scale of z, parameters: ok")


def randomize(model, film):
    with torch.no_grad():
        for n, p in list(model.named_parameters()) + [("film." + n, p) for n, p in film.named_parameters()]:
            if n.endswith(".B") or n == "film.net.2.weight":
                p.normal_(0, 0.1)


def test_export(base, d):
    from transformers import AutoModelForCausalLM
    x = torch.randint(0, 300, (2, 12))
    model, film, ctx = FM.build(base, 0, 4, 8, "cpu", grad_ckpt=False, film_z=8)
    assert film.codes is None and film.zdim == 8
    randomize(model, film)
    path = os.path.join(d, "z.pt")
    cfg = {"film_modes": 0, "film_z": 8, "lora_r": 4, "lora_alpha": 8, "lora_modes": 0}
    torch.save({"adapter": FM.adapter_state(model, film), "cfg": cfg}, path)
    shared = FM.export_shared(base, path, os.path.join(d, "shared"), dtype=torch.float32)
    assert {f for f in os.listdir(shared) if f.endswith(".safetensors")} == {FM.REST_SHARD, FM.NORM_SHARD}
    assert FM.INDEX in os.listdir(shared) and "config.json" in os.listdir(shared)
    lf = FM.load_film(cfg, torch.load(path, weights_only=False)["adapter"], len(FM.norm_modules(model)), 64)
    Z = film.sample_z(2, torch.Generator().manual_seed(3))
    model.eval()
    got = []
    for j in range(2):
        with torch.no_grad():
            out = FM.write_film_dir(shared, os.path.join(d, f"z{j}"), lf.gamma_z(Z[j:j + 1])[0])
            assert os.path.samefile(os.path.join(out, FM.REST_SHARD), os.path.join(shared, FM.REST_SHARD)), "a link"
            ctx.gamma = film.gamma_z(Z[j:j + 1].expand(2, -1))
            want = model(input_ids=x).logits
            got.append(AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32)(input_ids=x).logits)
        assert float((want - got[j]).abs().max()) < 1e-4, j
    assert float((got[0] - got[1]).abs().max()) > 1e-3, "different z must give different models"
    mcl = os.path.join(d, "mcl.pt")
    torch.save({"adapter": {}, "cfg": {"lora_modes": K, "lora_r": 4, "lora_alpha": 8}}, mcl)
    for fn in (lambda: FM.export_modes(base, path, [os.path.join(d, "bad")], [0]),     # z has no modes
               lambda: FM.export_shared(base, mcl, os.path.join(d, "bad"))):            # separate adapters
        try:
            fn()
            raise AssertionError("must be refused")
        except ValueError:
            pass
    # a FiLM code through the shared export gives the checkpoint of export_modes
    model, film, ctx = FM.build(base, K, 4, 8, "cpu", grad_ckpt=False)
    randomize(model, film)
    path = os.path.join(d, "codes.pt")
    cfg = {"film_modes": K, "lora_r": 4, "lora_alpha": 8}
    torch.save({"adapter": FM.adapter_state(model, film), "cfg": cfg}, path)
    shared = FM.export_shared(base, path, os.path.join(d, "shared_codes"))
    lf = FM.load_film(cfg, torch.load(path, weights_only=False)["adapter"], len(FM.norm_modules(model)), 64)
    for m in (1, 3):
        a = FM.write_film_dir(shared, os.path.join(d, f"code{m}"), lf.gamma(torch.tensor([m]))[0])
        b = FM.export_modes(base, path, [os.path.join(d, f"mode{m}")], [m])[0]
        wa = AutoModelForCausalLM.from_pretrained(a, dtype=torch.bfloat16).state_dict()
        wb = AutoModelForCausalLM.from_pretrained(b, dtype=torch.bfloat16).state_dict()
        assert wa.keys() == wb.keys() and all(torch.equal(wa[k], wb[k]) for k in wa), m
    print("checkpoints of a z and of a code: ok")


def test_train(d, p1, base):
    ex = [{"kind": "solve"}, {"kind": "solve"}, {"kind": "verify"}, {"kind": "verify"}]
    L = torch.tensor([[1.0, 3.0, 0.1, 0.1], [2.0, 0.5, 0.2, 0.0]])
    import random
    assert assign("wtaz", ex, L, 2, 0.0, random.Random(0)) == [0, 1, 0, 1], "argmin, no capacity for checks"
    from transformers import AutoTokenizer
    from phase1.train import load_jsonl, make_steps
    tok = AutoTokenizer.from_pretrained(base)
    tr, ver = load_jsonl(f"{p1}/data/train.jsonl"), load_jsonl(f"{p1}/data/verify.jsonl")
    sz, s1 = (make_steps(v, tr, ver, tok, 4, 0.4, 0) for v in ("wtaz", "wta1"))
    assert sz == s1 and sum(len(s) for s in sz) < sum(len(s) for s in make_steps("wtamulti", tr, ver, tok, 4, 0.4, 0)), \
        "wtaz trains on the examples of wta1: one reference per problem"

    def train(out, *extra, check=True):
        return subprocess.run([sys.executable, "-m", "phase1.train", "--variant", "wtaz", "--zdim", "8", "--base", base,
                               "--data", f"{p1}/data", "--out", out, "--P", "2", "--max_steps", "2", "--lora_r", "4",
                               *extra], env=ENV, capture_output=True, text=True, check=check)
    for name in ("a", "b"):
        train(os.path.join(d, f"wtaz_{name}"))
    res = json.load(open(os.path.join(d, "wtaz_a", "result.json")))
    assert res["zdim"] == 8 and res["z_gain"] >= 0 and res["mode_share"] is None and res["steps_done"] == 2, res
    ck = [torch.load(os.path.join(d, f"wtaz_{n}", "adapter.pt"), map_location="cpu", weights_only=False)
          for n in ("a", "b")]
    assert ck[0]["cfg"]["film_z"] == 8 and ck[0]["cfg"]["film_modes"] == 0
    assert not any("codes" in k for k in ck[0]["adapter"]) and "film.net.0.weight" in ck[0]["adapter"]
    assert ck[0]["adapter"]["film.net.0.weight"].shape == (64, 8)
    assert all(torch.equal(ck[0]["adapter"][k], ck[1]["adapter"][k]) for k in ck[0]["adapter"]), "same seed, same z"
    r = train(os.path.join(d, "wtaz_init"), "--init", os.path.join(d, "wtaz_a", "adapter.pt"), check=False)
    assert r.returncode != 0 and "from scratch" in r.stderr
    r = subprocess.run([sys.executable, "-m", "phase1.evaluate", "--base", base, "--adapter",
                        os.path.join(d, "wtaz_a", "adapter.pt"), "--data", f"{p1}/data", "--out", os.path.join(d, "ev")],
                       env=ENV, capture_output=True, text=True)
    assert r.returncode != 0 and "phase1.keval" in r.stderr, r.stderr
    print("wtaz training: ok")


def brute_pass(C, k):
    """Exact P(some correct) for k attempts spread over the strata as in zreport.pass_at."""
    S, n, N = C.shape
    base, r = divmod(k, S)
    tot, cnt = np.zeros(N), 0
    for R in itertools.combinations(range(S), r):
        picks = [list(itertools.combinations(range(n), base + (s in R))) for s in range(S)]
        for choice in itertools.product(*picks):
            tot += np.any([C[s, list(c)].any(0) if c else np.zeros(N, bool) for s, c in enumerate(choice)], axis=0)
            cnt += 1
    return tot / cnt


def test_estimators():
    rng = np.random.default_rng(0)
    for S, n, ks in ((1, 6, range(1, 7)), (4, 3, (1, 2, 3, 4, 5, 8, 12))):
        C = rng.random((S, n, 40)) < rng.random(40)
        for k in ks:
            assert np.allclose(zreport.pass_at(C, k), brute_pass(C, k)), (S, n, k)
    A = np.array([[[1, 2]], [[1, 3]], [[2, 3]], [[1, -1]]])                 # [S=4, n=1, N=2]
    truth = np.array([1, 3])
    got = zreport.maj_at(A, truth, 4, rng)
    assert list(got) == [1.0, 1.0], got                                       # 1 wins 3:1; 3 wins 2:1
    A = np.array([[[5], [5], [6], [-1]]])                                     # one stratum, 4 samples
    m2 = zreport.maj_at(A, np.array([5]), 2, np.random.default_rng(1), draws=4000)
    # pairs (6 equally likely): {5,5} -> 5; {5,6}: tie, earliest of the draw -> 1/2; {5,-1} -> 5; {6,-1} -> 6; {6,5}...
    assert abs(float(m2[0]) - (1 + 2 * 0.5 + 2) / 6) < 0.03, m2
    assert list(zreport.vote(np.array([[1, 4], [2, 4], [-1, 5]]), np.array([1, 5]))) == [0.5, 0.0], "a tie: 1/2"
    A = np.array([[[2]], [[1]]])                                              # modes 0 and 1 tie at k = 2
    assert list(zreport.maj_at(A, np.array([1]), 2, rng)) == [0.5], "mode 0 is not preferred"
    print("pass@k and maj@k estimators: ok")


def fake_refs(p1, seeds_run, base, real):
    """Trained references of seed 0: tiny real adapters (real=True) or placeholder files."""
    for name, d, kw in (("wta1", f"{p1}/train/wta1", {"film_modes": K}), ("sftmulti", f"{p1}/train/sftmulti", {}),
                        ("loramcl", f"{seeds_run}/train/loramcl_s0", {"lora_modes": K})):
        os.makedirs(d, exist_ok=True)
        json.dump({"steps_done": 400, "epochs_done": 0.5, "tokens": 13e6, "hours": 2.0, "trainable_M": 42.5},
                  open(f"{d}/result.json", "w"))
        if not real:
            open(f"{d}/adapter.pt", "w").close()
            continue
        model, film, _ = FM.build(base, kw.get("film_modes", 0), 4, 8, "cpu", grad_ckpt=False,
                                  lora_modes=kw.get("lora_modes", 0))
        if film is not None:
            randomize(model, film)
        cfg = {"film_modes": 0, "lora_r": 4, "lora_alpha": 8, "lora_modes": 0, "prefix_modes": 0, **kw}
        torch.save({"adapter": FM.adapter_state(model, film), "cfg": cfg}, f"{d}/adapter.pt")


def z_cmd(run, p1, seeds_run, base, *extra):
    return [sys.executable, "-m", "phase1.zrun", "--run", run, "--phase1", p1, "--seeds_run", seeds_run,
            "--model", base, *extra]


def test_plan(d, p1, base):
    seeds_run = os.path.join(d, "seeds_plan")
    p1b = os.path.join(d, "p1_plan")
    os.makedirs(f"{p1b}/data")
    for f in ("train", "verify"):
        open(f"{p1b}/data/{f}.jsonl", "w").write(open(f"{p1}/data/{f}.jsonl").read())
    with open(f"{p1b}/data/test.jsonl", "w") as f:              # the size of the real test: the estimates use it
        f.writelines(json.dumps({"id": f"t{i}", "question": "q", "answer": "7"}) + "\n" for i in range(1500))
    run = os.path.join(d, "plan")
    r = subprocess.run(z_cmd(run, p1b, seeds_run, base, "--plan_only"), env=ENV, capture_output=True, text=True)
    assert r.returncode != 0 and "loramcl_s0/adapter.pt" in r.stderr, r.stderr
    fake_refs(p1b, seeds_run, base, real=False)
    out = subprocess.run(z_cmd(run, p1b, seeds_run, base, "--plan_only"), env=ENV, capture_output=True, text=True,
                         check=True).stdout
    print(out.splitlines()[0])
    assert "12 jobs" in out and "k = 32, greedy 8" in out, out
    hours = float(out.split("simulated ")[1].split(" h")[0])
    assert 4.5 <= hours <= 5.5, hours
    assert "keval_wtaz_d1024: est 1.67 h, deps ['train_wtaz_d1024']" in out, out
    assert "keval_loramcl: est 1.22 h, deps []" in out, out
    from types import SimpleNamespace
    from phase1 import zrun
    a = SimpleNamespace(smoke=False, k=32, greedy=8, max_tokens=1200, phase1=p1b, seeds_run=seeds_run)
    J = {j.name: j.cmd for j in zrun.jobs(a, run, f"{p1b}/data", base)}
    c = J["train_wtaz_d128"]
    assert c[c.index("--variant") + 1] == "wtaz" and c[c.index("--zdim") + 1] == "128" and c[c.index("--seed") + 1] == "0"
    assert c[c.index("--max_hours") + 1] == "2.00" and "--lora_r" not in c, "r = 32, alpha 64 as wta1"
    c = J["keval_wta1"]
    assert c[c.index("--adapter") + 1] == f"{p1b}/train/wta1/adapter.pt" and c[c.index("--k") + 1] == "32"
    assert c[c.index("--seed") + 1] == "17", "the sampling seed of phase 1"
    json.dump({"k": 16}, open(f"{run}/state.json", "w"))
    r = subprocess.run(z_cmd(run, p1b, seeds_run, base, "--plan_only"), env=ENV, capture_output=True, text=True)
    assert r.returncode != 0 and "k = 16" in r.stderr, r.stderr
    print("plan: ok")


def test_smoke(d, p1, base):
    seeds_run = os.path.join(d, "seeds_run")
    fake_refs(p1, seeds_run, base, real=True)
    run = os.path.join(d, "smoke")
    r = subprocess.run(z_cmd(run, p1, seeds_run, base, "--smoke", "--max_tokens", "8", "--gpus", "0,1"), env=ENV)
    assert r.returncode == 0, "smoke run failed, see " + os.path.join(run, "logs")
    for zd in (8, 16, 128, 1024):
        e = f"{run}/eval/wtaz_d{zd}"
        man = json.load(open(f"{e}/eval.json"))
        assert man["kind"] == "z" and man["k"] == 4 and man["units"] == 4 and man["greedy"] == [0, 1], man
        assert tuple(torch.load(f"{e}/z.pt").shape) == (4, zd) and not os.path.exists(f"{e}/tmp_export")
        assert sorted(f for f in os.listdir(e) if f.startswith("greedy_")) == ["greedy_0_test.jsonl.gz",
                                                                             "greedy_1_test.jsonl.gz"]
    for name, kind, units, n in (("wta1", "modes", 4, 1), ("loramcl", "modes", 4, 1), ("sftmulti", "single", 1, 4)):
        man = json.load(open(f"{run}/eval/{name}/eval.json"))
        assert (man["kind"], man["units"], man["n"]) == (kind, units, n), (name, man)
    text = open(f"{run}/report_z.md", encoding="utf-8").read()
    print("\n".join(text.splitlines()[:14]))
    for row in ("| wtaz_d8 (", "| wtaz_d1024 (", "| wta1 (", "| loramcl (", "| sftmulti (", "| wtaz_d16 − wta1 |",
                "| wtaz_d1024 − wtaz_d16 |", "| wtaz_d128 | 128 | 2 |", "| wta1 | коды |"):
        assert row in text, row
    assert "нет данных" not in text and "не выполнены" not in text, text
    summ = json.load(open(f"{run}/summary_z.json"))
    assert summ["kinds"]["wtaz_d16"] == "z" and all(0 <= v <= 4 for m in summ["summary"].values() for v in m.values())
    # a restart with another sampling seed is refused instead of mixing the outputs
    man = json.load(open(f"{run}/eval/wtaz_d8/keval.json"))
    r = subprocess.run([sys.executable, "-m", "phase1.keval", "--base", base, "--adapter", man["adapter"], "--data",
                        os.path.dirname(man["test"]), "--out", f"{run}/eval/wtaz_d8", "--k", "4", "--greedy", "2",
                        "--seed", "99", "--max_tokens", "8"], env=ENV, capture_output=True, text=True)
    assert r.returncode != 0 and "'seed': (17, 99)" in r.stderr, r.stderr
    print("smoke run end to end: ok")


def main():
    d = tempfile.mkdtemp()
    base = tiny_model(d)
    test_scale_params()
    test_export(base, d)
    _, p1 = fake_phases(d)
    test_train(d, p1, base)
    test_estimators()
    test_plan(d, p1, base)
    test_smoke(d, p1, base)
    print("all ok")


if __name__ == "__main__":
    main()
