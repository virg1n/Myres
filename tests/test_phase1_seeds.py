"""Seeds run on CPU with a tiny random Qwen3 and fabricated phase-0 / phase-1 outputs (~5-10 min).

    PYTHONUTF8=1 python3 -m tests.test_phase1_seeds

1. separate adapters: a mixed batch equals single-mode batches, every exported mode equals training
2. report: known numbers, pairing (identical methods give exactly zero), seed 0 from phase 1
   rejected files (ids out of order, wrong number of verifier scores), direct pairs with the near methods
3. plan of the full run: all jobs, the simulated time
4. schedule by the tokens of another run; ModC data refuses a gap in the pools
5. the --smoke run end to end through the real job runner (train, evaluate on the HF backend, report)
6. resuming that run long after its start: new deadline, report rebuilt with the new results"""
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

from phase0 import prompts
from phase1 import film as FM
from phase1 import seeds_report
from phase1.metrics import load_eval
from phase1.train import mcl_weights
from tests.test_phase1_export import tiny_qwen

ENV = dict(os.environ, PYTHONUTF8="1", TOKENIZERS_PARALLELISM="false")
N_TEST = 90
K = 4


def write_gz(path, rows):
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def tiny_model(d):
    from transformers import AutoTokenizer
    base = os.path.join(d, "base")
    tiny_qwen(base)
    tok = AutoTokenizer.from_pretrained(base)
    tok.chat_template = "{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}{% endfor %}<assistant>"
    tok.save_pretrained(base)
    return base


def test_adapters(base, d):
    from transformers import AutoModelForCausalLM
    model, film, ctx = FM.build(base, 0, 8, 16, "cpu", grad_ckpt=False, lora_modes=K)
    assert film is None
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith(".B"):
                p.normal_(0, 0.1)
    path = os.path.join(d, "mcl.pt")
    torch.save({"adapter": FM.adapter_state(model, film), "cfg": {"film_modes": 0, "lora_r": 8, "lora_alpha": 16,
                                                                 "lora_modes": K}}, path)
    dirs = FM.export_modes(base, path, [os.path.join(d, f"mcl{k}") for k in range(K)], list(range(K)), dtype=torch.float32)
    x = torch.randint(0, 300, (K, 12))
    model.eval()
    with torch.no_grad():
        ctx.modes = torch.arange(K)
        mixed = model(input_ids=x).logits
        for k in range(K):
            ctx.modes = torch.tensor([k])
            one = model(input_ids=x[k:k + 1]).logits
            assert float((mixed[k] - one[0]).abs().max()) < 1e-4, k
            b = AutoModelForCausalLM.from_pretrained(dirs[k], dtype=torch.float32)(input_ids=x[k:k + 1]).logits
            assert float((one - b).abs().max()) < 1e-4, k
        ctx.modes = torch.zeros(K, dtype=torch.long)
        assert float((model(input_ids=x).logits[1] - mixed[1]).abs().max()) > 1e-3, "adapters must differ"
    q = mcl_weights(torch.tensor([[3.0, 1.0, 2.0, 4.0]]), 0.06)
    assert abs(float(q.sum()) - 1) < 1e-6 and abs(float(q[0, 1]) - 0.94) < 1e-6 and abs(float(q[0, 0]) - 0.02) < 1e-6
    assert "[Mode 2]" in prompts.solve_prompt(FakeTok(), "q", "neutral", 1)
    assert "[Mode" not in prompts.solve_prompt(FakeTok(), "q", "neutral")
    assert "[Mode 4]" in prompts.verify_prompt(FakeTok(), "q", "s", "neutral", 3)
    print("separate adapters and prefixes: ok")


class FakeTok:
    def apply_chat_template(self, msgs, **_):
        return " ".join(m["content"] for m in msgs)


# ----------------------------------------------------------------------------- fabricated outputs
def problems(n):
    return [{"id": f"t{i}", "source": "synth", "bin": f"synth_{i % 3}", "question": f"q{i}", "answer": "7"} for i in range(n)]


def write_modes(d, test, cor, wrong, verify=True):
    """cor [M,2,N] bool, wrong [M,2,N] int: an evaluation directory of a model with modes."""
    os.makedirs(d, exist_ok=True)
    M = cor.shape[0]
    for k in range(M):
        write_gz(os.path.join(d, f"solve_{k}_test.jsonl.gz"), [
            {"id": p["id"], "mode": "neutral", "texts": [f"{k}-{n}-{s}" for s in range(2)],
             "answers": ["7" if cor[k, s, n] else str(wrong[k, s, n]) for s in range(2)],
             "correct": [bool(cor[k, s, n]) for s in range(2)]} for n, p in enumerate(test)])
    if verify:
        for j in range(M):
            json.dump({"p_no": [[float(not cor[i, 0, n]) * 0.8 + 0.1 for n in range(len(test))] for i in range(M)],
                       "missing": 0}, open(os.path.join(d, f"verify_{j}.json"), "w"))
    json.dump({"kind": "modes", "modes": M, "sets": ["test"], "verify": verify}, open(os.path.join(d, "eval.json"), "w"))


def write_single(d, test, cor, wrong):
    """cor [8,N] bool: an evaluation directory of a model without modes."""
    os.makedirs(d, exist_ok=True)
    write_gz(os.path.join(d, "solve_single_test.jsonl.gz"), [
        {"id": p["id"], "mode": "neutral", "texts": [f"{s}-{n}" for s in range(8)],
         "answers": ["7" if cor[s, n] else str(wrong[s, n]) for s in range(8)],
         "correct": [bool(cor[s, n]) for s in range(8)]} for n, p in enumerate(test)])
    json.dump({"p_no": [[float(not cor[i, n]) * 0.8 + 0.1 for n in range(len(test))] for i in range(4)], "missing": 0},
              open(os.path.join(d, "verify_single.json"), "w"))
    json.dump({"kind": "single", "modes": 4, "sets": ["test"], "verify": True}, open(os.path.join(d, "eval.json"), "w"))


def write_phase0(d, test, rng):
    for sub in ("data", "test", "pool"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    write_jsonl(os.path.join(d, "data", "test.jsonl"), test)
    N = len(test)
    cor = rng.random((K, 2, N)) < 0.6
    for k in range(K):
        write_gz(os.path.join(d, "test", f"test_strategy{k}.jsonl.gz"), [
            {"id": p["id"], "mode": k, "texts": ["a", "b"], "answers": ["7" if cor[k, s, n] else "9" for s in range(2)],
             "correct": [bool(cor[k, s, n]) for s in range(2)]} for n, p in enumerate(test)])
    mat = [[[0.5] * N for _ in range(K)] for _ in range(K)]
    json.dump({"p_no": mat, "missing": 0}, open(os.path.join(d, "test", "verify_strategies.json"), "w"))
    write_gz(os.path.join(d, "test", "test_neutral.jsonl.gz"), [
        {"id": p["id"], "mode": "neutral", "texts": ["a"] * 8, "answers": ["7"] * 8, "correct": [True] * 8} for p in test])
    json.dump({"p_no": [[0.5] * N for _ in range(K)], "missing": 0}, open(os.path.join(d, "test", "verify_neutral.json"), "w"))
    return cor


def test_report(d):
    rng = np.random.default_rng(1)
    test = problems(N_TEST)
    run, p1, p0 = (os.path.join(d, x) for x in ("rep_run", "rep_p1", "rep_p0"))
    os.makedirs(os.path.join(p1, "data"))
    os.makedirs(run)
    write_jsonl(os.path.join(p1, "data", "test.jsonl"), test)
    p0_cor = write_phase0(p0, test, rng)
    data = {}
    for s in (0, 1, 2):
        cor = rng.random((K, 2, N_TEST)) < 0.7 + 0.05 * s
        wrong = rng.integers(100, 103, (K, 2, N_TEST))
        data[s] = cor
        write_modes(os.path.join(p1, "eval", "wta1") if s == 0 else os.path.join(run, "eval", f"wta1_s{s}"), test, cor, wrong)
        write_modes(os.path.join(p1, "eval", "wtabal") if s == 0 else os.path.join(run, "eval", f"wtabal_s{s}"), test, cor, wrong)
        one = rng.random((8, N_TEST)) < 0.6
        w8 = rng.integers(100, 103, (8, N_TEST))
        write_single(os.path.join(p1, "eval", "sftmulti") if s == 0 else os.path.join(run, "eval", f"sftmulti_s{s}"), test, one, w8)
    # no seed 2 for this one: comparisons must use the seeds that both methods have
    for s in (0, 1):
        write_single(os.path.join(run, "eval", f"sftmulti_eq_s{s}"), test, rng.random((8, N_TEST)) < 0.5,
                     rng.integers(100, 103, (8, N_TEST)))
    out = seeds_report.build(run, os.path.join(p1, "data"), p1, p0, boot=50)
    s = json.load(open(os.path.join(run, "summary_seeds.json")))
    want = np.mean([data[x][:, 0].mean() for x in (0, 1, 2)])
    assert abs(s["summary"]["wta1"]["pass1"]["mean"] - want) < 1e-9
    want4 = [data[x][:, 0].any(0).mean() for x in (0, 1, 2)]
    assert abs(s["summary"]["wta1"]["pass4"]["mean"] - np.mean(want4)) < 1e-9
    assert abs(s["summary"]["wta1"]["pass4"]["sd"] - np.std(want4, ddof=1)) < 1e-9
    assert abs(s["summary"]["p0_strategies"]["pass1"]["mean"] - p0_cor[:, 0].mean()) < 1e-9
    assert abs(s["summary"]["wta1"]["auroc_own"] - 1.0) < 1e-9
    # phi is lower-is-better but the difference is always X - R; X and p0 come with their own seeds
    c = s["comparisons"]
    d0 = c["wta1-p0_strategies:pass1"]
    assert d0["n"] == 3 and d0["lo"] <= d0["mean"] <= d0["hi"]
    assert c["wta1-sftmulti_eq:pass1"]["n"] == 2, "only common seeds"
    assert "sftmulti-sftmulti_eq:pass1" in c and "sftmulti_eq-sftmulti:pass1" not in c
    text = open(out, encoding="utf-8").read()
    assert "wtabal seed" not in text and "sftmulti_eq seed 2" in text
    assert "loramcl seed 1" in text, "missing methods are listed"
    # pairing: a method compared with itself gives exactly 0 in every bootstrap replicate
    A = seeds_report.load_all(run, p1, p0, test)[0]
    pt, rp = {}, {}
    for name in ("wta1", "wtabal"):
        for x, a in A[name].items():
            vec = seeds_report.vectors(a)
            idxs = [np.random.default_rng(i).integers(0, N_TEST, N_TEST) for i in range(30)]
            pt[name, x] = {k: seeds_report.metric(a, vec, k, np.arange(N_TEST)) for k, _ in seeds_report.METRICS}
            rp[name, x] = {k: np.array([seeds_report.metric(a, vec, k, i) for i in idxs]) for k, _ in seeds_report.METRICS}
    for k, _ in seeds_report.METRICS:
        e = seeds_report.compare(pt, rp, "wta1", "wtabal", [(x, x) for x in (0, 1, 2)], k)
        assert e["mean"] == 0 and e["lo"] == 0 and e["hi"] == 0, k
    # direct pairs with the near methods, for our variants only
    for s_ in (1, 2):
        write_modes(os.path.join(run, "eval", f"loramcl_s{s_}"), test, data[s_], np.full((K, 2, N_TEST), 100))
        write_modes(os.path.join(run, "eval", f"modc_s{s_}"), test, data[s_], np.full((K, 2, N_TEST), 100))
        write_single(os.path.join(run, "eval", f"sftmodc_s{s_}"), test, rng.random((8, N_TEST)) < 0.5, np.full((8, N_TEST), 100))
        write_single(os.path.join(run, "eval", f"sft1_eq_s{s_}"), test, rng.random((8, N_TEST)) < 0.5, np.full((8, N_TEST), 100))
    # a report must refuse results that do not follow the test set instead of mixing up problems
    f = os.path.join(run, "eval", "wtabal_s2", "solve_1_test.jsonl.gz")
    rows = [json.loads(l) for l in gzip.open(f, "rt")]
    write_gz(f, [rows[1], rows[0]] + rows[2:])
    f = os.path.join(run, "eval", "wta1_s2", "verify_0.json")
    v = json.load(open(f))
    json.dump({"p_no": [x[:-1] for x in v["p_no"]], "missing": 0}, open(f, "w"))
    try:
        load_eval(os.path.join(run, "eval", "wtabal_s2"), test, "test")
        raise AssertionError("rows out of order must be rejected")
    except ValueError as e:
        assert "solve_1_test" in str(e)
    out = seeds_report.build(run, os.path.join(p1, "data"), p1, p0, boot=10)
    c = json.load(open(os.path.join(run, "summary_seeds.json")))["comparisons"]
    text = open(out, encoding="utf-8").read()
    assert "wtabal seed 2 (отклонён" in text and "wta1 seed 2 (отклонён" in text
    for pair in ("wta1-loramcl", "wta1-modc", "wtabal-loramcl", "wtabal-modc", "modc-sftmodc", "wta1-sft1_eq"):
        assert f"{pair}:pass4" in c, pair
    assert c["wta1-loramcl:pass1"]["n"] == 1, "wta1 seed 2 was rejected, seed 0 of loramcl does not exist"
    for pair in ("loramcl-modc", "sftmodc-modc", "wta1-sftmodc", "sft1_eq-sftmulti", "loramcl-wta1"):
        assert f"{pair}:pass4" not in c, pair
    print("report on fabricated data: ok")


# ----------------------------------------------------------------------------- full pipeline
def fake_phases(d):
    """Stubs of the outputs of phases 0 and 1 (train/verify from a labelled pool)."""
    p0, p1 = os.path.join(d, "p0"), os.path.join(d, "p1")
    for sub in (f"{p1}/data", f"{p1}/hard/out", f"{p0}/pool", f"{p0}/data", f"{p0}/test"):
        os.makedirs(sub, exist_ok=True)
    tr = problems(30)
    train = [dict(p, refs=[f"hello {p['id']} a", f"hello {p['id']} b"], single=f"hello {p['id']} a", rate=0.5) for p in tr]
    write_jsonl(f"{p1}/data/train.jsonl", train)
    write_jsonl(f"{p1}/data/verify.jsonl", [{"question": p["question"], "solution": "hello", "label": "Yes" if i % 2 else "No"}
                                            for i, p in enumerate(tr)])
    write_jsonl(f"{p1}/data/test.jsonl", problems(30))
    pool = []
    for k in range(K):
        for i, p in enumerate(tr):
            pool.append({"id": p["id"], "mode": k, "texts": [f"strategy{k} {p['id']}", "x"], "answers": ["7", "9"],
                         "correct": [i % (k + 1) == 0, False]})
    write_gz(f"{p0}/pool/pool_0.jsonl.gz", pool[:len(pool) // 2])
    write_gz(f"{p1}/hard/out/pool_0.jsonl.gz", pool[len(pool) // 2:])
    return p0, p1


def seeds_cmd(run, p0, p1, base, *extra):
    return [sys.executable, "-m", "phase1.seeds_run", "--run", run, "--phase0", p0, "--phase1", p1, "--model", base,
            "--gpus", "0,1", *extra]


def test_plan(d, p0, p1, base):
    missing = subprocess.run(seeds_cmd(os.path.join(d, "plan"), p0, p1, base, "--plan_only"), env=ENV,
                             capture_output=True, text=True)
    assert missing.returncode != 0 and "wta1/result.json" in missing.stderr, missing.stderr
    os.makedirs(f"{p1}/train/wta1")
    json.dump({"tokens": 16000000, "time_limit_h": 2.0}, open(f"{p1}/train/wta1/result.json", "w"))
    for v in ("wta1", "wtabal", "sftmulti"):
        os.makedirs(f"{p1}/eval/{v}")
        open(f"{p1}/eval/{v}/eval.json", "w").write("{}")
    for f in ["verify_strategies.json", "verify_neutral.json"] + [f"test_strategy{k}.jsonl.gz" for k in range(K)]:
        open(f"{p0}/test/{f}", "w").close()
    r = subprocess.run(seeds_cmd(os.path.join(d, "plan"), p0, p1, base, "--plan_only", "--gpus", "0,1,2,3"),
                       env=ENV, capture_output=True, text=True, check=True)
    out = r.stdout
    print(out.splitlines()[0])
    assert "42 jobs" in out, out
    for name in ("train_loramcl_s0", "train_modc_s0", "train_sftmulti_eq_s0", "train_wta1_s1", "train_sftmulti_s2",
                 "eval_sftmulti_eq_s2", "eval_modc_s0", "train_sft1_eq_s0", "eval_sft1_eq_s2", "train_sftmodc_s0",
                 "eval_sftmodc_s1"):
        assert name + ":" in out, name
    for name in ("train_wta1_s0", "train_wtabal_s0", "train_sftmulti_s0", "eval_wta1_s0"):
        assert name + ":" not in out, name
    assert "train_sft1_eq_s1: est 0.90 h, deps ['train_wta1_s1']" in out, "own seed of wta1"
    assert "train_sft1_eq_s0: est 0.90 h, deps []" in out, "wta1 of seed 0 comes from phase 1"
    hours = float(out.split("simulated ")[1].split(" h")[0])
    assert hours <= 14, hours
    print("plan of the full run: ok")


def test_target_and_gaps(d, p0, p1, base):
    # the run ends at the tokens of the other run, not at its time limit or at the end of the data
    ref = os.path.join(d, "ref.json")
    json.dump({"tokens": 60}, open(ref, "w"))
    out = os.path.join(d, "target")
    subprocess.run([sys.executable, "-m", "phase1.train", "--variant", "sft1", "--base", base, "--data", f"{p1}/data",
                    "--out", out, "--target_tokens_from", ref, "--P", "2", "--seed", "3", "--save_opt", "0"],
                   env=ENV, check=True, capture_output=True)
    r = json.load(open(f"{out}/result.json"))
    assert r["tokens"] >= 60 and r["steps_done"] == 1 and r["seed"] == 3, r
    # a train problem without a labelled reference is an error, not a silent skip
    gap = os.path.join(d, "gap")
    os.makedirs(gap)
    write_gz(os.path.join(gap, "pool_0.jsonl.gz"), [{"id": "t0", "mode": 0, "texts": ["a"], "answers": ["7"], "correct": [True]}])
    x = subprocess.run([sys.executable, "-m", "phase1.modc_data", "--data", f"{p1}/data", "--pools", gap, "--out",
                        os.path.join(d, "gap_out")], env=ENV, capture_output=True, text=True)
    assert x.returncode != 0 and "1 of 30" in x.stderr, x.stderr
    print("schedule by tokens, gaps in the pools: ok")


def test_smoke(d, p0, p1, base):
    run = os.path.join(d, "smoke")
    r = subprocess.run(seeds_cmd(run, p0, p1, base, "--smoke", "--max_tokens", "8"), env=ENV)
    assert r.returncode == 0, "smoke run failed, see " + os.path.join(run, "logs")
    res = {v: json.load(open(f"{run}/train/{v}/result.json")) for v in ("loramcl_s1", "modc_s1", "wta1_s1", "sftmulti_s1")}
    assert res["loramcl_s1"]["seed"] == 1 and sum(res["loramcl_s1"]["mode_share"]) > 0.99
    assert res["modc_s1"]["mode_share"] is not None and res["sftmulti_s1"]["mode_share"] is None
    assert os.path.exists(f"{run}/train/sftmulti_s1/snap/adapter.pt"), "snapshot inside a full run"
    assert not os.path.exists(f"{run}/train/sftmulti_eq_s0/snap/adapter.pt"), "the stopped run has adapter.pt only"
    assert json.load(open(f"{run}/train/sftmulti_eq_s0/snap/snap.json"))["step"] == 1
    assert res["sftmulti_s1"]["steps_done"] == 2
    ck = torch.load(f"{run}/train/loramcl_s1/adapter.pt", map_location="cpu", weights_only=False)
    assert ck["cfg"]["lora_modes"] == K and "opt" not in ck
    a = next(v for n, v in ck["adapter"].items() if n.endswith(".A"))
    assert a.shape[0] == K
    b = [v for n, v in ck["adapter"].items() if n.endswith(".B")][0]
    assert float((b[0] - b[1]).abs().max()) > 0, "every adapter must receive gradient"
    ck = torch.load(f"{run}/train/modc_s1/adapter.pt", map_location="cpu", weights_only=False)
    assert ck["cfg"]["prefix_modes"] == K and ck["cfg"]["film_modes"] == 0
    a1 = torch.load(f"{run}/train/wta1_s1/adapter.pt", map_location="cpu", weights_only=False)["adapter"]
    a0 = torch.load(f"{run}/train/sftmulti_eq_s0/adapter.pt", map_location="cpu", weights_only=False)["adapter"]
    key = next(n for n in a0 if n.endswith(".A"))
    assert float((a0[key] - a1[key]).abs().max()) > 0, "seed must change the initialization"
    with open(f"{run}/modc/train.jsonl") as f:
        rows = [json.loads(l) for l in f]
    assert all(r["refs"] and all(x["text"].startswith(f"strategy{x['mode']}") for x in r["refs"]) for r in rows)
    for v, s in (("loramcl", 1), ("modc", 1), ("wta1", 1), ("sftmulti", 1), ("sftmulti_eq", 0), ("sftmulti_eq", 1)):
        assert os.path.exists(f"{run}/eval/{v}_s{s}/DONE"), (v, s)
        assert not os.path.exists(f"{run}/eval/{v}_s{s}/tmp_export"), "checkpoints must be deleted"
    assert os.path.exists(f"{run}/eval/modc_s1/verify_1.json")
    text = open(f"{run}/report_seeds.md", encoding="utf-8").read()
    print("\n".join(text.splitlines()[:14]))
    assert "| loramcl (" in text and "| modc (" in text and "| sft1_eq (" in text
    assert "### Против loramcl" in text and "### Против modc" in text and "### Против sft1_eq" in text
    assert "wta1-loramcl:pass1" in json.load(open(f"{run}/summary_seeds.json"))["comparisons"]
    r1 = json.load(open(f"{run}/train/sft1_eq_s1/result.json"))
    assert r1["tokens"] >= res["wta1_s1"]["tokens"] and r1["seed"] == 1
    assert os.path.exists(f"{run}/train/sftmodc_s1/DONE")
    print("smoke run end to end: ok")

    # a day later, with an evaluation that was lost: not skipped by the old deadline, report rebuilt
    sp = f"{run}/state.json"
    st = json.load(open(sp))
    st["started"] -= 24 * 3600
    json.dump(st, open(sp, "w"))
    assert os.path.exists(f"{run}/report_job/DONE")
    shutil.rmtree(f"{run}/eval/loramcl_s1")
    text = text.replace("| loramcl (", "")
    open(f"{run}/report_seeds.md", "w", encoding="utf-8").write(text)
    r = subprocess.run(seeds_cmd(run, p0, p1, base, "--smoke", "--max_tokens", "8", "--max_hours", "4"), env=ENV)
    assert r.returncode == 0
    assert os.path.exists(f"{run}/eval/loramcl_s1/DONE")
    assert "| loramcl (" in open(f"{run}/report_seeds.md", encoding="utf-8").read(), "the report must be rebuilt"
    print("resume after the deadline of the first start: ok")


def main():
    d = tempfile.mkdtemp()
    base = tiny_model(d)
    test_adapters(base, d)
    test_report(d)
    p0, p1 = fake_phases(d)
    test_plan(d, p0, p1, base)
    test_target_and_gaps(d, p0, p1, base)
    test_smoke(d, p0, p1, base)
    print("all ok")


if __name__ == "__main__":
    main()
