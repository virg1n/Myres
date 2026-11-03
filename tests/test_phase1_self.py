"""Multi-round self-training on CPU with a tiny random Qwen3 and fabricated phases 0 / 1 (~5-10 min).

    PYTHONUTF8=1 python3 -m tests.test_phase1_self

1. schedule of the sets per round, attempts per mode, sampling seeds, strata of the base screening
2. bank: 4 slots with the latest admitted solution, \\boxed{} required, weights of rft / rftbal / modes, frozen ignored
3. training on a bank from the start adapter: FiLM codes, separate adapters (a row reaches only its own adapter),
   no modes; the number of steps
4. pool: 0-correct problems of the old pools, fresh synthetic ones, screening strata, disjoint hv / pool, frozen
5. report estimators: complementarity of modes, first attempts, the frozen curve
6. plan: jobs, shared round 0, priorities, swap, a changed setting refused
7. the --smoke run end to end through the real job runner (HF backend), then the report"""
import gzip
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

from phase1 import selftrain as ST
from phase1 import spool, sreport, srun
from tests.test_phase1_seeds import ENV, tiny_model
from tests.test_phase1_z import fake_refs

K = 4


def write_gz(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_units():
    assert ST.parse_sets("pool:4,hv:16") == [("pool", 4), ("hv", 16)]
    assert ST.mode_sets([("pool", 4), ("hv", 16), ("frozen", 20)], 4) == "pool:1,hv:4,frozen:5"
    assert ST.mode_sets([("pool", 4), ("hv", 16)], 0) == "pool:4,hv:16"
    try:
        ST.mode_sets([("hv", 6)], 4)
        raise AssertionError("6 attempts over 4 modes must be refused")
    except SystemExit:
        pass
    R = 6
    assert srun.sets(0, R) == {"pool": 4, "hv": 16, "test": 4, "frozen": 20}
    assert srun.sets(1, R) == {"pool": 4} and srun.sets(2, R) == {"pool": 4, "hv": 4, "test": 4}
    assert srun.sets(5, R) == {"pool": 4} and srun.sets(R, R) == {"hv": 16, "test": 4}
    assert srun.sets(3, 3) == {"hv": 16, "test": 4} and srun.sets(2, 3) == {"pool": 4, "hv": 4, "test": 4}
    seeds = {srun.gen_seed(s, r, w) + 1000 * (m + 1) for s in (0, 1) for r in range(7)
             for w in ("rft", "rftbal", "wta1", "loramcl", "swap", "sftmulti") for m in range(4)}
    assert len(seeds) == 2 * 7 * 6 * 4, "every mode of every generation step samples with its own seed"
    assert [spool.stratum(k, 8) for k in (0, 1, 2, 3, 6, 7, 8)] == ["B0", "B12", "B12", "B36", "B36", None, None]
    from phase0 import answers
    assert answers.key(r"\frac10") == r"\frac10" and answers.key(r"\frac12") == "1/2", "a malformed answer, no crash"
    import random
    rows = [{"stratum": st, "source": "s"} for st in ("A", "B0", "B12", "x", "y", "z") for _ in range(5)]
    for n in (4, 7, 10):
        take, rest = spool.split(list(rows), n, random.Random(0))
        assert len(take) == n and len(take) + len(rest) == len(rows), "exactly n held out"
    print("schedule, seeds, strata: ok")


def problems(n, prefix="p", stratum="B0"):
    return [{"id": f"{prefix}{i}", "source": "synth", "bin": "synth_9", "question": f"question {prefix}{i}",
             "answer": "7", "stratum": stratum} for i in range(n)]


def fake_gen(d, probs, M, texts_of, name="pool"):
    """A generation directory: texts_of(problem index, mode) -> list of (text, correct)."""
    os.makedirs(d, exist_ok=True)
    for m in range(max(M, 1)):
        rows = []
        for j, p in enumerate(probs):
            tc = texts_of(j, m)
            rows.append({"id": p["id"], "mode": "neutral", "texts": [t for t, _ in tc],
                         "answers": ["7" if c else "3" for _, c in tc], "correct": [c for _, c in tc]})
        write_gz(os.path.join(d, f"solve_{m}_{name}.jsonl.gz"), rows)
    gj = os.path.join(d, "gen.json")
    man = json.load(open(gj)) if os.path.exists(gj) else {"modes": M, "sets": {}, "seed": 0, "adapter": ""}
    man["sets"][name] = len(texts_of(0, 0)) * max(M, 1)
    json.dump(man, open(gj, "w"))
    open(os.path.join(d, "DONE"), "w").close()


def test_bank(d):
    probs = problems(4)
    pd = {p["id"]: p for p in probs}
    bx = lambda r, m: f"round {r} mode {m} so \\boxed{{7}}"
    # modes: p0 solved by mode 1 in round 0 and again in round 2 (newer text kept), by mode 3 in round 1;
    # p1 right without \boxed (not admitted); p2 never; p3 by all 4 modes in round 1
    def modes_round(r):
        def f(j, m):
            ok = (j == 0 and ((m == 1 and r in (0, 2)) or (m == 3 and r == 1))) or (j == 3 and r == 1)
            if j == 1 and m == 0:
                return [("the answer is 7", True)]
            return [(bx(r, m) if ok else "wrong \\boxed{3}", ok)]
        return f
    gd = [os.path.join(d, "bank_modes", f"r{r}") for r in range(3)]
    for r in range(3):
        fake_gen(gd[r], probs, K, modes_round(r))
        fake_gen(gd[r], probs, K, lambda j, m: [(bx(9, m), True)], name="frozen")   # never part of a bank
    rows = ST.build_bank(gd, pd, "modes")
    assert [r["id"] for r in rows] == ["p0", "p3"], rows
    p0 = rows[0]
    assert [(s["slot"], s["round"], s["mode"]) for s in p0["sols"]] == [(1, 2, 1), (3, 1, 3)] and p0["first"] == 0
    assert p0["sols"][0]["solution"] == bx(2, 1) and all(s["w"] == 0.5 for s in p0["sols"])
    assert len(rows[1]["sols"]) == 4 and all(s["w"] == 0.25 for s in rows[1]["sols"]) and rows[1]["first"] == 1
    st = ST.bank_stats(rows, 2)
    assert st["problems"] == 2 and st["solutions"] == 6 and st["full_share"] == 0.5 and st["new_last_round"] == 0
    assert st["per_slot"] == [1, 2, 1, 2]
    # no modes: slots are the 4 samples of a round; rft weighs every solution 1, rftbal each problem 1
    gs = [os.path.join(d, "bank_single", f"r{r}") for r in range(2)]
    fake_gen(gs[0], probs, 0, lambda j, m: [(bx(0, s), j == 0 and s in (0, 2)) for s in range(4)])
    fake_gen(gs[1], probs, 0, lambda j, m: [(bx(1, s), j == 0 and s == 2 or j == 2 and s == 3) for s in range(4)])
    for kind, w in (("rft", 1.0), ("rftbal", 0.5)):
        rows = ST.build_bank(gs, pd, kind)
        assert [r["id"] for r in rows] == ["p0", "p2"], (kind, rows)
        assert [(s["slot"], s["round"]) for s in rows[0]["sols"]] == [(0, 0), (2, 1)] and rows[0]["sols"][0]["w"] == w
        assert all(s["mode"] is None for r in rows for s in r["sols"]) and rows[1]["first"] == 1
    swap = ST.build_bank(gd, pd, "swap")
    assert all(s["mode"] is None for r in swap for s in r["sols"]) and swap[0]["sols"][0]["w"] == 0.5
    print("bank: ok")
    return gd, gs


def adapter(base, path, film_modes=0, lora_modes=0):
    from phase1 import film as FM
    model, film, _ = FM.build(base, film_modes, 4, 8, "cpu", grad_ckpt=False, lora_modes=lora_modes)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith(".B"):
                p.normal_(0, 0.1)
    cfg = {"variant": "x", "film_modes": film_modes, "lora_r": 4, "lora_alpha": 8, "lora_modes": lora_modes,
           "prefix_modes": 0, "film_z": 0}
    torch.save({"adapter": FM.adapter_state(model, film), "cfg": cfg}, path)
    return path


def test_train(d, base):
    data = os.path.join(d, "tdata")
    probs = problems(6)
    write_jsonl(os.path.join(data, "pool.jsonl"), probs)
    only1 = lambda j, m: [("hello 123 so \\boxed{7}", m == 1)]           # every problem solved by mode 1 only
    g = os.path.join(d, "tgen_modes")
    fake_gen(g, probs, K, only1)
    gs = os.path.join(d, "tgen_single")
    fake_gen(gs, probs, 0, lambda j, m: [("hello world \\boxed{7}", s < 2) for s in range(4)])

    def run(name, init, gens, kind, *extra):
        out = os.path.join(d, name)
        r = subprocess.run([sys.executable, "-m", "phase1.selftrain", "train", "--base", base, "--init", init, "--gens",
                            gens, "--kind", kind, "--data", data, "--out", out, "--P", "2", "--steps_max", "50",
                            "--epochs", "1", "--lr", "1e-2", *extra], env=ENV, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        return out, torch.load(os.path.join(out, "adapter.pt"), weights_only=False)

    mcl0 = adapter(base, os.path.join(d, "mcl0.pt"), lora_modes=K)
    out, ck = run("t_mcl", mcl0, g, "modes")
    st, res = json.load(open(f"{out}/bank.json")), json.load(open(f"{out}/result.json"))
    assert st["steps"] == 3 == res["steps_done"], "min(steps_max, 1 pass over 6 problems of 2 per step)"
    assert res["mode_share"] == [0.0, 1.0, 0.0, 0.0] and ck["cfg"]["lora_modes"] == K and ck["cfg"]["film_modes"] == 0
    ref = torch.load(mcl0, weights_only=False)["adapter"]
    assert all(v.shape == ref[n].shape for n, v in ck["adapter"].items())
    changed = [m for m in range(K) if any(not torch.equal(v[m], ref[n][m].float()) for n, v in ck["adapter"].items())]
    assert changed == [1], f"only the adapter of mode 1 may change, changed {changed}"
    film0 = adapter(base, os.path.join(d, "film0.pt"), film_modes=K)
    out, ck = run("t_film", film0, g, "modes")
    ref = torch.load(film0, weights_only=False)["adapter"]
    codes = ck["adapter"]["film.codes"]
    assert torch.equal(codes[0], ref["film.codes"][0].float()) and not torch.equal(codes[1], ref["film.codes"][1].float())
    assert ck["cfg"]["film_modes"] == K and ck["cfg"]["variant"] == "fixed"
    one = adapter(base, os.path.join(d, "one.pt"))
    for kind in ("rft", "rftbal"):
        out, ck = run(f"t_{kind}", one, gs, kind)
        b = [json.loads(l) for l in open(f"{out}/bank.jsonl")]
        assert len(b) == 6 and all(len(r["sols"]) == 2 for r in b)
        assert {s["w"] for r in b for s in r["sols"]} == ({1.0} if kind == "rft" else {0.5})
        assert any(not torch.equal(v, torch.load(one, weights_only=False)["adapter"][n].float())
                   for n, v in ck["adapter"].items()), "trained"
    out, ck = run("t_swap", one, g, "swap")
    assert ck["cfg"]["film_modes"] == 0 and ck["cfg"]["lora_modes"] == 0
    empty = os.path.join(d, "tgen_empty")
    fake_gen(empty, probs, 0, lambda j, m: [("x", False)] * 4)
    out, ck = run("t_empty", one, empty, "rft")
    assert json.load(open(f"{out}/result.json"))["steps_done"] == 0, "an empty bank trains nothing"
    print("training on the bank (FiLM codes, separate adapters, no modes): ok")


def fake_inputs(d):
    """Phases 0 / 1 with 0-correct problems: p0 pool g0..g9 (g0-g3 never solved), hard pool h0..h5 (h0, h1 never)."""
    p0, p1 = os.path.join(d, "p0"), os.path.join(d, "p1")
    g = [{"id": f"g{i}", "source": "gsm8k", "bin": "gsm8k", "question": f"pool question {i}", "answer": "7"}
         for i in range(10)]
    h = [{"id": f"h{i}", "source": "math", "bin": "math_L5", "question": f"hard question {i}", "answer": "7"}
         for i in range(6)]
    test = [{"id": f"t{i}", "source": "synth", "bin": "synth_5", "question": f"test question {i}", "answer": "7"}
            for i in range(12)]
    write_jsonl(f"{p0}/data/pool.jsonl", g)
    write_jsonl(f"{p0}/data/test.jsonl", test)
    write_jsonl(f"{p1}/hard/pool.jsonl", h)
    write_gz(f"{p0}/pool/pool_0.jsonl.gz", [{"id": p["id"], "correct": [i >= 4] * 8} for i, p in enumerate(g)])
    write_gz(f"{p1}/hard/out/pool_0.jsonl.gz", [{"id": p["id"], "correct": [i >= 2, False] * 4} for i, p in enumerate(h)])
    write_jsonl(f"{p1}/data/train.jsonl", [dict(p, refs=["r"], single="r") for p in g[4:] + h[2:]])
    write_jsonl(f"{p1}/data/val.jsonl", [])
    # g3 was a verifier example ("No"): Av; h1 has the question of g5 (trained under another id): out
    write_jsonl(f"{p1}/data/verify.jsonl", [{"question": "pool question 3", "solution": "s", "label": "No"}])
    h[1]["question"] = g[5]["question"]
    write_jsonl(f"{p1}/hard/pool.jsonl", h)
    write_jsonl(f"{p1}/data/test.jsonl", test)
    return p0, p1


def test_pool(d):
    p0, p1 = fake_inputs(d)
    run = os.path.join(d, "pool_run")
    base = ["--phase0", p0, "--phase1", p1, "--out", run, "--shards", "2"]
    subprocess.run([sys.executable, "-m", "phase1.spool", "cands", *base, "--n_math", "0", "--n_synth", "40",
                    "--ops", "3", "5"], env=ENV, check=True, capture_output=True)
    A = [json.loads(l) for l in open(f"{run}/pool/A.jsonl")]
    assert [(p["id"], p["stratum"]) for p in A] == [("g0", "A"), ("g1", "A"), ("g2", "A"), ("g3", "Av"), ("h0", "A")], A
    cands = [json.loads(l) for s in range(2) for l in open(f"{run}/pool/cand/shard{s}/test.jsonl")]
    assert len(cands) == 40 and all(p["id"].startswith("synth-self-") for p in cands)
    # screening: candidate i solved i % 9 times of 8 (8 -> dropped as easy)
    for s in range(2):
        rows = [json.loads(l) for l in open(f"{run}/pool/cand/shard{s}/test.jsonl")]
        write_gz(f"{run}/pool/screen/shard{s}/solve_0_test.jsonl.gz",
                 [{"id": p["id"], "correct": [k < int(p["id"].split("-")[-1]) % 9 for k in range(8)]} for p in rows])
    subprocess.run([sys.executable, "-m", "phase1.spool", "select", *base, "--n_hv", "6", "--n_hard", "100",
                    "--n_med", "5", "--n_frozen", "4"], env=ENV, check=True, capture_output=True)
    P = {n: [json.loads(l) for l in open(f"{run}/data/{n}.jsonl")] for n in ("pool", "hv", "frozen", "test")}
    ids = {n: {p["id"] for p in rows} for n, rows in P.items()}
    assert not ids["pool"] & ids["hv"] and ids["frozen"] <= ids["pool"] and len(P["test"]) == 12
    assert all(p["stratum"] in ("A", "B0", "B12") for p in P["hv"]) and "g3" in ids["pool"], "Av: pool only"
    assert all(p["stratum"] in ("A", "Av", "B0", "B12") for p in P["frozen"])
    assert sum(p["stratum"] == "B36" for p in P["pool"]) == 5 and len(P["hv"]) == 6
    assert all(int(p["id"].split("-")[-1]) % 9 < 7 for p in P["pool"] + P["hv"] if p["id"].startswith("synth"))
    hard = len(A) + sum(int(p["id"].split("-")[-1]) % 9 <= 2 for p in cands)
    assert sum(p["stratum"] != "B36" for p in P["pool"]) == hard - 6
    # the screening must have 8 attempts of every candidate
    write_gz(f"{run}/pool/screen/shard0/solve_0_test.jsonl.gz", [{"id": cands[0]["id"], "correct": [False] * 7}])
    r = subprocess.run([sys.executable, "-m", "phase1.spool", "select", *base], env=ENV, capture_output=True, text=True)
    assert r.returncode != 0 and "8 attempts" in r.stderr, r.stderr
    meta = json.load(open(f"{run}/data/meta.json"))
    assert meta["sizes"]["hv"] == 6 and meta["pool"]["B36"] == 5
    print("pool: ok")
    return p0, p1


def test_estimators():
    C = np.zeros((4, 4, 3), dtype=bool)
    C[0, :, 0] = True                        # problem 0: only mode 0 (always); problem 1: modes 1 and 2 once each
    C[1, 0, 1] = C[2, 3, 1] = True
    E = {"C": C, "A": np.where(C, 0, 1), "X": np.ones_like(C), "truth": np.zeros(3, dtype=int)}
    cm = sreport.complementarity(E)
    assert abs(cm["within4"] - (1 + 1 + 1 + 0) / 3 / 4) < 1e-9 and abs(cm["balanced4"] - (1 + (1 - 0.75 * 0.75)) / 3) < 1e-9
    assert abs(cm["exclusive"] - 0.5) < 1e-9 and cm["per_mode_acc"][0] == 1 / 3
    v = sreport.vectors(E, np.random.default_rng(0))
    assert list(v["distinct4"]) == [2, 2, 1] and set(v) >= {"pass1", "pass4", "pass16", "maj4", "noboxed"}
    print("report estimators: ok")


def self_cmd(run, p0, p1, seeds_run, base, *extra):
    return [sys.executable, "-m", "phase1.srun", "--run", run, "--phase0", p0, "--phase1", p1, "--seeds_run",
            seeds_run, "--model", base, *extra]


def test_plan(d, p0, p1, base):
    seeds_run = os.path.join(d, "seeds_plan")
    run = os.path.join(d, "plan")
    r = subprocess.run(self_cmd(run, p0, p1, seeds_run, base, "--plan_only"), env=ENV, capture_output=True, text=True)
    assert r.returncode != 0 and "loramcl_s0/adapter.pt" in r.stderr, r.stderr
    fake_refs(p1, seeds_run, base, real=False)
    for v in ("wta1", "sftmulti", "loramcl"):
        os.makedirs(f"{seeds_run}/train/{v}_s1", exist_ok=True)
        open(f"{seeds_run}/train/{v}_s1/adapter.pt", "w").close()
    out = subprocess.run(self_cmd(run, p0, p1, seeds_run, base, "--plan_only"), env=ENV, capture_output=True,
                         text=True, check=True).stdout
    print(out.splitlines()[0])
    # 1 + 4 + 1 pool jobs; per seed 3 round-0 generations, 4 chains x 6 rounds x (train + gen), swap 2; 2 reports
    assert f"{1 + 4 + 1 + 2 * (3 + 48 + 2) + 2} jobs" in out, out
    hours = float(out.split("simulated ")[1].split(" h")[0])
    alone = float(out.split("alone ")[1].split(" h")[0])
    assert 8 <= hours <= 15 and 4 <= alone <= 8, (hours, alone)
    line = lambda name: next(l for l in out.splitlines() if f"  {name}:" in l)
    assert "deps ['gen_s0_sftmulti_r0']" in line("train_s0_rft_r1") and "deps ['gen_s0_sftmulti_r0']" in line("train_s0_rftbal_r1")
    assert "prio 20" in line("gen_s0_wta1_r3") and "prio 10" in line("gen_s1_wta1_r3") and "prio 20" in line("gen_s0_swap")
    assert "deps ['gen_s0_wta1_r5']" in line("train_s0_swap")
    a = srun.parse(["--phase0", p0, "--phase1", p1, "--seeds_run", seeds_run])
    J = {j.name: j.cmd for j in srun.jobs(a, run, base)}
    c = J["train_s0_wta1_r3"]
    assert c[c.index("--init") + 1] == f"{p1}/train/wta1/adapter.pt", "every round restarts from the start adapter"
    assert c[c.index("--gens") + 1].split(",") == [f"{run}/s0/start_wta1/gen", f"{run}/s0/wta1/r1/gen",
                                                    f"{run}/s0/wta1/r2/gen"]
    c = J["train_s0_swap"]
    assert c[c.index("--init") + 1] == f"{p1}/train/sftmulti/adapter.pt" and c[c.index("--kind") + 1] == "swap"
    assert c[c.index("--gens") + 1].split(",")[0] == f"{run}/s0/start_wta1/gen"
    c = J["gen_s1_loramcl_r6"]
    assert c[c.index("--sets") + 1] == "hv:16,test:4" and c[c.index("--drop_adapter") + 1] == "0"
    c = J["gen_s1_loramcl_r2"]
    assert c[c.index("--sets") + 1] == "pool:4,hv:4,test:4" and c[c.index("--drop_adapter") + 1] == "1"
    c = J["gen_s1_sftmulti_r0"]
    assert c[c.index("--adapter") + 1] == f"{seeds_run}/train/sftmulti_s1/adapter.pt"
    assert c[c.index("--sets") + 1] == "pool:4,hv:16,test:4,frozen:20"
    st = json.load(open(f"{run}/state.json")) if os.path.exists(f"{run}/state.json") else {}
    json.dump({**st, "rounds": 4}, open(f"{run}/state.json", "w"))
    r = subprocess.run(self_cmd(run, p0, p1, seeds_run, base, "--plan_only"), env=ENV, capture_output=True, text=True)
    assert r.returncode != 0 and "rounds = 4" in r.stderr, r.stderr
    print("plan: ok")


def test_smoke(d, p0, p1, base):
    seeds_run = os.path.join(d, "seeds_run")
    p1s = os.path.join(d, "p1s")
    import shutil
    shutil.copytree(p1, p1s)
    fake_refs(p1s, seeds_run, base, real=True)
    run = os.path.join(d, "smoke")
    r = subprocess.run(self_cmd(run, p0, p1s, seeds_run, base, "--smoke", "--max_tokens", "8", "--gpus", "0,1"),
                       env=ENV)
    assert r.returncode == 0, "smoke run failed, see " + os.path.join(run, "logs")
    for c in srun.CHAINS:
        for rr in (1, 2):
            assert os.path.exists(f"{run}/s0/{c}/r{rr}/train/bank.json")
        assert not os.path.exists(f"{run}/s0/{c}/r1/train/adapter.pt"), "intermediate adapters are dropped"
        assert os.path.exists(f"{run}/s0/{c}/r2/train/adapter.pt"), "the final adapter is kept"
        g = json.load(open(f"{run}/s0/{c}/r2/gen/gen.json"))
        assert g["sets"] == {"hv": 16, "test": 4} and g["modes"] == (0 if c.startswith("rft") else K)
    g = json.load(open(f"{run}/s0/start_wta1/gen/gen.json"))
    assert g["sets"] == {"pool": 4, "hv": 16, "test": 4, "frozen": 20} and g["modes"] == K
    assert os.path.exists(f"{run}/s0/start_wta1/gen/solve_3_frozen.jsonl.gz")
    assert not os.path.exists(f"{run}/s0/start_sftmulti/gen/solve_1_pool.jsonl.gz"), "no modes: one checkpoint"
    text = open(f"{run}/report_self.md", encoding="utf-8").read()
    print("\n".join(text.splitlines()[:6]))
    for row in ("## 1. Главная метрика", "| wta1 − rftbal (главное) |", "| wta1 (wta1: 4 кода FiLM", "| swap (",
                "| frozen: wta1 без обучения |", "| s0 loramcl M_2 |", "| s0 wta1 | ", "## 8. Банк wta1"):
        assert row in text, row
    assert "нет финальной модели" not in text and "не выполнены" not in text and "отклонены" not in text, text
    summ = json.load(open(f"{run}/summary_self.json"))
    assert summ["complete_seeds"] == [0] and "primary:wta1-rftbal" in summ["comparisons"]
    assert os.path.exists(f"{run}/audit_sample.md") and os.path.exists(f"{run}/audit_key.json")
    # rft and rftbal share round 0: the same numbers; a planned seed without results makes the verdict preliminary
    m0 = {c: summ["summary"][f"s0:{c}:M0:hv"] for c in ("rft", "rftbal")}
    assert m0["rft"] == m0["rftbal"], m0
    sreport.build(run, [0, 1], list(srun.CHAINS), 2, boot_n=20)
    text = open(f"{run}/report_self.md", encoding="utf-8").read()
    assert "предварительно: нет seed [1]" in text and "нет финальной модели:** s1:rftbal" in text
    assert not json.load(open(f"{run}/summary_self.json"))["comparisons"]["primary:wta1-rftbal"]["win"]
    print("smoke run end to end: ok")


def main():
    d = tempfile.mkdtemp()
    base = tiny_model(d)
    test_units()
    test_bank(d)
    test_train(d, base)
    p0, p1 = test_pool(d)
    test_estimators()
    test_plan(d, p0, p1, base)
    test_smoke(d, p0, p1, base)
    print("all ok")


if __name__ == "__main__":
    main()
