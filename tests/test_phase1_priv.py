"""Private-share sweep on CPU with a tiny random Qwen3 / Llama and fabricated phase-0 / phase-1 outputs (~5-10 min).

    PYTHONUTF8=1 python3 -m tests.test_phase1_priv

1. the model: shared + private LoRA of mode k is the rank r_s + r_p LoRA [B_s B_k][A_s; A_k]; a mixed batch equals
   single-mode batches; an exported mode equals the training model; adapters of phase1.train (wta1, LoRA-MCL, SFT)
   convert to the same function; a private LoRA without modes is refused
2. training: sft / wta / mcl on math and mt run, save the priv format, change only the parameters they should (wta: a
   step changes the private LoRA of the assigned modes only), the budget by tokens and by epochs, the LR schedules,
   a last.pt of another configuration is refused
3. references, NLL (equals a row-by-row computation), report statistics on known inputs (NMI, best match, split rates)
4. translation: the reference file of fairseq, the prompt / target split, ALMA's output cleaning, the metrics on
   hand-checkable inputs, evaluation of a tiny Llama (modes and the beam baseline without modes)
5. plans of both benchmarks (points, reused adapters, refused inputs) and the math --smoke run end to end through the
   real job runner (train, generate on the HF backend, NLL, report)"""
import gzip
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

from phase1 import film as FM
from phase1 import pmt, preport, priv as PV, ptrain
from tests.test_phase1_seeds import ENV, tiny_model, write_gz, write_jsonl

K = 4


def randomize(model, film, std=0.1):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith((".B_s", ".B_p", ".B")):
                p.normal_(0, std)
        if film is not None:
            film.net[2].weight.normal_(0, std)


def test_model(base, d):
    from transformers import AutoModelForCausalLM
    torch.manual_seed(0)
    model, film, ctx = PV.build(base, K, 3, 2, 10.0, True, "cpu", grad_ckpt=False)
    randomize(model, film)
    model.eval()
    lin = model.model.layers[0].self_attn.q_proj
    for k in range(K):             # the per-mode adapter is one rank-5 LoRA
        A = torch.cat([lin.A_s, lin.A_p[k]]).detach()
        B = torch.cat([lin.B_s, lin.B_p[k]], 1).detach()
        assert torch.allclose(lin.delta(k), 2.0 * B @ A, atol=1e-6)
    x = torch.randint(0, 300, (K, 10))
    with torch.no_grad():
        ctx.modes = torch.arange(K)
        ctx.gamma = film.gamma(ctx.modes)
        mixed = model(input_ids=x).logits
        ck = {"adapter": PV.adapter_state(model, film), "cfg": {"format": "priv", "K": K, "r_s": 3, "r_p": 2,
                                                               "alpha": 10.0, "film": True}}
        for k in range(K):
            PV.set_mode(ctx, film, k)
            one = model(input_ids=x[k:k + 1]).logits
            assert float((mixed[k] - one[0]).abs().max()) < 1e-4
            out = PV.export_mode(base, ck, os.path.join(d, "exp"), k, dtype=torch.float32)
            ex = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32)(input_ids=x[k:k + 1]).logits
            assert float((one - ex).abs().max()) < 1e-4, f"export of mode {k}"
    # adapters of phase1.train convert exactly: wta1 (FiLM + LoRA), LoRA-MCL (separate adapters), SFT
    for fm, lm in ((K, 0), (0, K), (0, 0)):
        torch.manual_seed(1)
        om, of, oc = FM.build(base, fm, 4, 8, "cpu", grad_ckpt=False, lora_modes=lm)
        with torch.no_grad():
            for n, p in om.named_parameters():
                if n.endswith(".B"):
                    p.normal_(0, 0.1)
            if of is not None:
                of.net[2].weight.normal_(0, 0.1)
        om.eval()
        old = {"adapter": FM.adapter_state(om, of), "cfg": {"film_modes": fm, "lora_r": 4, "lora_alpha": 8,
                                                            "lora_modes": lm, "variant": "x"}}
        nm, nf, nc, c = PV.build_from(base, old, "cpu")
        assert (c["r_s"], c["r_p"], c["film"]) == ((0, 4, False) if lm else (4, 0, bool(fm)))
        nm.eval()
        with torch.no_grad():
            for k in (range(K) if fm or lm else [None]):
                if k is None:
                    oc.modes, oc.gamma = None, None
                else:
                    oc.modes = torch.tensor([k])
                    oc.gamma = of.gamma(oc.modes) if of is not None else None
                PV.set_mode(nc, nf, k)
                assert float((om(input_ids=x[:1]).logits - nm(input_ids=x[:1]).logits).abs().max()) < 1e-5
    try:
        PV.build(base, 1, 4, 2, 8.0, False, "cpu", grad_ckpt=False)
        raise AssertionError("private LoRA with one mode accepted")
    except ValueError:
        pass
    from transformers import AutoConfig
    p = PV.n_params({"K": K, "r_s": 3, "r_p": 2, "film": True}, AutoConfig.from_pretrained(base))
    have = {s: sum(v.numel() for n, v in model.named_parameters() if n.endswith(s)) for s in ("_s", "_p")}
    assert (p["shared"], p["private"], p["film"]) == (have["_s"], have["_p"], sum(v.numel() for v in film.parameters()))
    print("model: rank, mixed batch, export, conversion of old adapters: ok")


def fake_phase(d, base):
    """phase0 / phase1 / phase1_self / phase1_seeds with a few short problems."""
    p0, p1, sr, se = (os.path.join(d, x) for x in ("phase0", "phase1", "self", "seeds"))
    for x in (f"{p0}/pool", f"{p1}/data", f"{sr}/data", f"{se}/modc"):
        os.makedirs(x, exist_ok=True)
    tr = [{"id": f"t{i}", "source": "synth", "question": f"what is {i} + 1", "answer": str(i + 1),
           "refs": [f"{i} + 1 = \\boxed{{{i + 1}}}"], "single": f"so {i + 1} \\boxed{{{i + 1}}}", "rate": 0.5}
          for i in range(30)]
    write_jsonl(f"{p1}/data/train.jsonl", tr)
    write_jsonl(f"{p1}/data/verify.jsonl", [{"question": "what is 1 + 1", "solution": "\\boxed{2}", "label": "Yes"},
                                             {"question": "what is 1 + 1", "solution": "\\boxed{3}", "label": "No"}] * 6)
    val = [{"id": f"v{i}", "source": "math" if i % 2 else "synth", "bin": "x", "question": f"what is {i} * 2",
            "answer": str(2 * i)} for i in range(60)]
    write_jsonl(f"{p1}/data/val.jsonl", val)
    write_gz(f"{p0}/pool/pool_0.jsonl.gz", [{"id": v["id"], "mode": k, "texts": [f"{k} way {v['answer']}", "x"],
                                             "answers": [v["answer"], "1"], "correct": [True, False]}
                                            for v in val for k in range(4) if (k + int(v["id"][1:])) % 3])
    json.dump({"model_path": base}, open(f"{p1}/state.json", "w"))
    write_jsonl(f"{sr}/data/hv.jsonl", [{"id": f"h{i}", "source": "synth", "question": f"what is {i} - 1",
                                         "answer": str(i - 1)} for i in range(8)])
    write_jsonl(f"{se}/modc/train.jsonl", [{"id": r["id"], "source": "synth", "question": r["question"],
                                            "refs": [{"mode": 0, "text": r["refs"][0]}, {"mode": 2, "text": r["single"]}]}
                                           for r in tr])
    return p0, p1, sr, se


def run_train(base, data, out, *extra, task="math"):
    ptrain.main(["--task", task, "--base", base, "--data", data, "--out", out, "--ckpt_minutes", "0", *extra])
    return torch.load(os.path.join(out, "adapter.pt"), weights_only=False), json.load(open(os.path.join(out, "result.json")))


def test_train(base, d, p1):
    data = f"{p1}/data"
    ck, res = run_train(base, data, f"{d}/tr_sft", "--variant", "sft", "--r_s", "4", "--r_p", "0", "--film", "0",
                        "--max_steps", "2", "--P", "4")
    assert ck["cfg"]["format"] == "priv" and ck["cfg"]["K"] == 1 and res["steps_done"] == 2
    assert all(k.endswith((".A_s", ".B_s")) for k in ck["adapter"])
    ck, res = run_train(base, data, f"{d}/tr_wta", "--variant", "wta", "--r_s", "3", "--r_p", "1", "--max_steps", "3",
                        "--P", "8", "--lr", "1e-2")
    assert ck["cfg"]["r_p"] == 1 and any(k.startswith("film.") for k in ck["adapter"]) and res["mode_share"]
    B = [v for k, v in ck["adapter"].items() if k.endswith(".B_p")][0]
    assert all(float(B[k].abs().sum()) > 0 for k in range(K) if res["mode_share"][k] > 0), "assigned modes move"
    ck, res = run_train(base, data, f"{d}/tr_mcl", "--variant", "mcl", "--r_s", "0", "--r_p", "4", "--film", "0",
                        "--max_steps", "2", "--P", "4")
    assert not any(k.startswith("film.") for k in ck["adapter"])
    B = [v for k, v in ck["adapter"].items() if k.endswith(".B_p")][0]
    assert all(float(B[k].abs().sum()) > 0 for k in range(K)), "relaxed WTA trains every adapter"
    # the budget by tokens ends the run at the first step past it
    _, res = run_train(base, data, f"{d}/tr_tok", "--variant", "wta", "--r_s", "2", "--r_p", "2",
                       "--target_tokens", "1500", "--P", "4")
    assert 1500 <= res["tokens"] < 1500 + 4000 and res["stop"] == "budget"
    try:
        ptrain.main(["--task", "math", "--variant", "sft", "--base", base, "--data", data, "--out", f"{d}/bad",
                     "--r_p", "2", "--film", "0", "--max_steps", "1"])
        raise AssertionError("sft with a private LoRA accepted")
    except SystemExit:
        pass
    # a last.pt of another configuration is refused
    os.makedirs(f"{d}/tr_res", exist_ok=True)
    torch.save({"cfg": {"format": "priv", "x": 1}}, f"{d}/tr_res/last.pt")
    try:
        ptrain.main(["--task", "math", "--variant", "sft", "--base", base, "--data", data, "--out", f"{d}/tr_res",
                     "--r_s", "4", "--film", "0", "--max_steps", "1"])
        raise AssertionError("foreign last.pt accepted")
    except SystemExit:
        pass
    # schedules
    # Hugging Face inverse sqrt with 2 warm-up steps: 0, 0.5, then 1 / sqrt(step / 2)
    assert [ptrain.lr_mult("invsqrt", i, 100, 0, 2) for i in (0, 1, 2)] == [0.0, 0.5, 1.0]
    assert abs(ptrain.lr_mult("invsqrt", 8, 100, 0, 2) - 0.5) < 1e-12
    # the safety time limit: no DONE, last.pt kept, the run is not finished
    try:
        ptrain.main(["--task", "math", "--variant", "sft", "--base", base, "--data", data, "--out", f"{d}/tr_lim",
                     "--r_s", "4", "--film", "0", "--max_steps", "50", "--P", "4", "--max_hours", "0"])
        raise AssertionError("time limit ignored")
    except SystemExit as e:
        assert "safety time limit" in str(e)
    assert os.path.exists(f"{d}/tr_lim/last.pt") and not os.path.exists(f"{d}/tr_lim/DONE")
    # another recipe in a used directory (here: another LR) is refused, also once the run is finished
    try:
        ptrain.main(["--task", "math", "--variant", "sft", "--base", base, "--data", data, "--out", f"{d}/tr_sft",
                     "--r_s", "4", "--r_p", "0", "--film", "0", "--max_steps", "2", "--P", "4", "--lr", "1e-3"])
        raise AssertionError("another recipe accepted")
    except SystemExit as e:
        assert "another recipe" in str(e)
    assert abs(ptrain.lr_mult("cos", 100, 0, 1.0, 30) - 0.1) < 1e-12
    assert abs(ptrain.eps_at(0.1, 0.3, 0.05) - 0.175) < 1e-12 and abs(ptrain.eps_at(0.5, 0.3, 0.05) - 0.05) < 1e-12
    print("training: sft / wta / mcl, budget by tokens, refused settings, schedules: ok")


def test_train_modc(base, d, p1, se):
    """oracle / match / the controls on strategy-labelled references (fake modc: strategies 0 and 2 per problem)."""
    from phase1.train import assign, make_steps
    from transformers import AutoTokenizer
    data, mf = f"{p1}/data", f"{se}/modc/train.jsonl"
    mc = ["--modc_file", mf, "--max_steps", "3"]
    ck, res = run_train(base, data, f"{d}/mc_oracle", "--variant", "oracle", "--r_s", "4", "--r_p", "0", *mc)
    assert res["refs"] == "modc" and res["P"] == 8 and ck["cfg"]["refs"] == "modc"     # P = round(16 / 2 refs)
    sh = res["mode_share"]
    assert sh[1] == 0 and sh[3] == 0 and abs(sh[0] - 0.5) < 1e-9 and abs(sh[2] - 0.5) < 1e-9, sh
    _, res = run_train(base, data, f"{d}/mc_match", "--variant", "match", "--r_s", "4", "--r_p", "0", *mc)
    assert abs(sum(res["mode_share"]) - 1) < 1e-5          # float32 shares
    _, res = run_train(base, data, f"{d}/mc_wta", "--variant", "wta", "--r_s", "4", "--r_p", "0", *mc)
    assert res["refs"] == "modc"
    ck, res = run_train(base, data, f"{d}/mc_sft", "--variant", "sft", "--r_s", "4", "--film", "0", *mc)
    assert ck["cfg"]["K"] == 1 and res["mode_share"] is None
    # recipe of a run without labelled references: no modc_file key (directories made before the option resume)
    assert "modc_file" not in json.load(open(f"{d}/tr_sft/recipe.json"))
    assert json.load(open(f"{d}/mc_sft/recipe.json"))["modc_file"] == os.path.abspath(mf)
    # match: the references of one problem always get different modes (eps 0)
    tok = AutoTokenizer.from_pretrained(base)
    rows = [json.loads(l) for l in open(mf)]
    ex = make_steps("sftmodc", rows, [], tok, 5, 0.0, 0, K)[0]
    assert {e["mode"] for e in ex} == {0, 2} and len(ex) == 10
    cap = math.ceil(len(ex) / K * 1.25)
    first = set()
    for t in range(40):
        L = torch.rand(K, len(ex), generator=torch.Generator().manual_seed(t))
        for eps in (0.0, 0.5, 1.0):        # exploration keeps distinct modes inside a problem
            modes = ptrain.match_assign(ex, L, K, eps, random.Random(t))
            for g in {e["group"] for e in ex}:
                ms = [modes[i] for i, e in enumerate(ex) if e["group"] == g]
                assert len(set(ms)) == len(ms), (t, eps, g, ms)
            if eps == 0:
                assert max(np.bincount(modes, minlength=K)) <= cap, "batch capacity of wta1"
        assert assign("fixed", ex, L, K, 0.5, random.Random(t)) == [e["mode"] for e in ex], "oracle: no eps on solutions"
        # equal losses (identical modes): the choice is random, not the order of the references (sorted by strategy)
        tie = ptrain.match_assign(ex, torch.zeros(K, len(ex)), K, 0.0, random.Random(t))
        first.add(tuple(tie[i] for i, e in enumerate(ex) if e["group"] == ex[0]["group"]))
    assert len(first) > 3, first
    # without capacity pressure the matching is the best one: two references, losses that prefer modes 3 and 1
    two = [dict(p=[1], t=[1], group=0, kind="solve", mode=0, w=1.0), dict(p=[1], t=[1], group=0, kind="solve", mode=2, w=1.0)]
    L = torch.tensor([[5.0, 5.0], [4.0, 0.0], [5.0, 5.0], [0.0, 4.0]])
    assert ptrain.match_assign(two, L, K, 0.0, random.Random(0)) == [3, 1]
    for bad, why in ((["--variant", "oracle"], "needs --modc_file"),
                     (["--variant", "oracle", "--modc_file", mf, "--K", "2"], "needs --K"),
                     (["--variant", "mcl", "--modc_file", mf, "--r_s", "0", "--r_p", "4", "--film", "0"], "for math")):
        try:
            ptrain.main(["--task", "math", "--base", base, "--data", data, "--out", f"{d}/mc_bad", "--max_steps", "1",
                         *bad])
            raise AssertionError(f"accepted: {bad}")
        except SystemExit as e:
            assert why in str(e), (bad, str(e))
    print("training on labelled references: oracle, match, controls, refused settings: ok")


def test_refs_nll(base, d, p0, p1, se):
    from phase1 import peval
    peval.main(["refs", "--phase0", p0, "--phase1", p1, "--modc", f"{se}/modc/train.jsonl", "--n_train", "5",
                "--out", f"{d}/refs"])
    rows = preport.read_gz(f"{d}/refs/refs.jsonl.gz")
    val = [r for r in rows if r["split"] == "val"]
    assert len(val) == sum(1 for i in range(60) for k in range(4) if (k + i) % 3)
    assert all(r["text"].startswith(f"{r['strategy']} way") for r in val), "first correct text of the strategy"
    assert sum(r["split"] == "train" for r in rows) == 10
    torch.manual_seed(3)
    model, film, ctx = PV.build(base, K, 2, 2, 8.0, True, "cpu", grad_ckpt=False)
    randomize(model, film)
    ad = f"{d}/nll_adapter.pt"
    torch.save({"adapter": PV.adapter_state(model, film), "cfg": {"format": "priv", "K": K, "r_s": 2, "r_p": 2,
                                                                 "alpha": 8.0, "film": True}}, ad)
    peval.main(["nll", "--base", base, "--adapter", ad, "--refs", f"{d}/refs/refs.jsonl.gz", "--out", f"{d}/nll"])
    D = preport.jload(f"{d}/nll/nll.json.gz")
    from transformers import AutoTokenizer
    from phase1.train import encode_solve
    tok = AutoTokenizer.from_pretrained(base)
    model.eval()
    for i in (0, 7):
        p, t = encode_solve(tok, rows[i]["question"], rows[i]["text"])
        x = torch.tensor([p + t])
        for k in (0, 3):
            PV.set_mode(ctx, film, k)
            with torch.no_grad():
                lp = torch.log_softmax(model(input_ids=x).logits[0].float(), -1)
            want = -sum(float(lp[len(p) + j - 1, t[j]]) for j in range(len(t)))
            assert abs(D["nll"][k][i] - want) < 1e-3 * max(1, abs(want)), (i, k, D["nll"][k][i], want)
    # statistics
    assert abs(preport.nmi([0, 0, 1, 1], [5, 5, 7, 7]) - 1) < 1e-12 and abs(preport.nmi([0, 1, 0, 1], [5, 5, 7, 7])) < 1e-12
    assert preport.best_match([0, 0, 1, 1, 2], ["a", "a", "b", "b", "b"], 3) == 0.8
    w, a = preport.split_rates([0, 0, 1, 1], ["p", "p", "q", "q"])
    assert w == 0 and a == 1, "modes that split problems: never within, always across"
    w, a = preport.split_rates([0, 1, 0, 1], ["p", "p", "q", "q"])
    assert w == 1 and a == 0.5
    st = preport.loss_stats(np.array([[1.0, 4.0], [2.0, 3.0]]), [1, 2], None, np.array([3.0, 3.0]), ["a", "a"])
    # per reference: NLL / tokens = [[1, 2], [2, 1.5]], SFT [3, 1.5]; pooled over tokens: best 4 / 3, SFT 2
    assert abs(st["min"] - 1.25) < 1e-12 and abs(st["mean"] - 1.625) < 1e-12 and abs(st["min_minus_sft"] + 1) < 1e-12
    assert abs(st["min_tok"] - 4 / 3) < 1e-12 and abs(st["min_tok_minus_sft"] + 2 / 3) < 1e-12
    assert st["win_share"] == [0.5, 0.5] and st["split_within"] == 1
    print("references, NLL against a row-by-row computation, statistics: ok")


def tiny_llama(path):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.train_from_iterator(["Translate this from English to German: the house das Haus"] * 50,
                           trainers.BpeTrainer(vocab_size=320, special_tokens=["<unk>", "<s>", "</s>"],
                                               initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>", unk_token="<unk>",
                            pad_token="<unk>").save_pretrained(path)
    cfg = LlamaConfig(vocab_size=320, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=1, bos_token_id=1, eos_token_id=2, pad_token_id=0)
    LlamaForCausalLM(cfg).save_pretrained(path)


def test_mt(d):
    base = os.path.join(d, "llama")
    tiny_llama(base)
    raw = os.path.join(d, "mtraw")
    os.makedirs(raw, exist_ok=True)
    with open(f"{raw}/en", "w", encoding="utf-8") as f:
        f.write("\n".join(f"the house {i}" for i in range(40)) + "\n")
    with open(f"{raw}/de", "w", encoding="utf-8") as f:
        f.write("\n".join(f"das Haus {i}" for i in range(40)) + "\n")
    with open(f"{raw}/test", "w", encoding="utf-8") as f:
        for i in range(5):
            f.write(f"S-{i}\tthe house @-@ {i}\nT-{i}\tdas Haus {i}\n")
            for r in range(10):
                f.write(f"R{r + 1}-{i}\tdas Haus {i} , {'ja' if r % 2 else 'nein'}\n")
    pmt.main(["data", "--out", f"{d}/mtdata", "--train_en", f"{raw}/en", "--train_de", f"{raw}/de", "--test_file", f"{raw}/test"])
    tr = preport.load_jsonl(f"{d}/mtdata/train.jsonl")
    te = preport.load_jsonl(f"{d}/mtdata/test.jsonl")
    assert len(tr) == 40 and len(te) == 5 and all(len(r["refs"]) == 10 for r in te) and te[1]["src"] == "the house @-@ 1"
    assert pmt.clean("\n  Das Haus.\nmore") == "Das Haus." and pmt.clean("Ja\nNein") == "Ja"
    assert pmt.detok(["das Haus , ja ."]) == ["das Haus, ja."]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base)
    p, t, ok = ptrain.encode_mt(tok, "the house", " das Haus")
    full = tok(ptrain.mt_prompt("the house") + " das Haus")["input_ids"]
    assert p + t[:-1] == full and t[-1] == tok.eos_token_id and ok
    # metrics on hand-checkable inputs
    H = [["das Haus ist rot"] * 3, ["das Haus ist rot"] * 3]
    assert abs(pmt.pairwise_bleu(H) - 100) < 1e-9, "identical hypotheses: Pairwise-BLEU 100"
    R = [["das Haus ist rot", "ein Haus"]] * 3
    m = pmt.metrics(H, R)
    assert abs(m["oracle_sentence_bleu"] - 100) < 1e-9 and m["refs_covered"] == 1.0 and abs(m["bleu_all_refs_mean"] - 100) < 1e-9
    H2 = [["das Haus ist rot"] * 3, ["ein Haus"] * 3]
    m2 = pmt.metrics(H2, R)
    assert m2["refs_covered"] == 2.0 and m2["pairwise_bleu"] < 50
    for v, extra in (("sft", ["--r_s", "4", "--r_p", "0", "--film", "0"]), ("wta", ["--K", "3", "--r_s", "3", "--r_p", "1"]),
                     ("mcl", ["--K", "3", "--r_s", "0", "--r_p", "4", "--film", "0"])):
        ck, res = run_train(base, f"{d}/mtdata", f"{d}/mt_{v}", "--variant", v, *extra, "--P", "8", "--epochs", "1",
                            "--mb_tokens", "512", task="mt")
        assert res["steps_done"] == 5 and abs(ck["cfg"]["dropout"] - 0.05) < 1e-12
        pmt.main(["eval", "--base", base, "--adapter", f"{d}/mt_{v}/adapter.pt", "--data", f"{d}/mtdata",
                  "--out", f"{d}/mte_{v}", "--beams", "1,2", "--K", "3", "--max_new", "6", "--bs", "4"])
        M = json.load(open(f"{d}/mte_{v}/metrics.json"))
        for b in ("1", "2"):
            hy = json.load(open(f"{d}/mte_{v}/hyps_b{b}.json"))
            assert len(hy) == 3 and all(len(h) == 5 for h in hy) and M["beams"][b]["K"] == 3
        D = preport.jload(f"{d}/mte_{v}/nll.json.gz")
        assert D["modes"] == (1 if v == "sft" else 3) and len(D["ntok"]) == 50
    print("translation: data, prompt split, cleaning, metrics, training and evaluation of 3 variants: ok")
    return base


def prun_cmd(run, p0, p1, sr, se, *extra):
    return [sys.executable, "-m", "phase1.prun", "--run", run, "--phase0", p0, "--phase1", p1, "--self_run", sr,
            "--seeds_run", se, "--gpus", "0", *extra]


def test_plan_and_smoke(d, base, p0, p1, sr, se):
    r = subprocess.run(prun_cmd(f"{d}/plan", p0, p1, sr, se, "--bench", "math", "--plan_only"), env=ENV,
                       capture_output=True, text=True)
    assert r.returncode != 0 and "loramcl_s0/adapter.pt" in r.stdout + r.stderr, "missing reused adapters are named"
    os.makedirs(f"{p1}/train/wta1", exist_ok=True)
    os.makedirs(f"{se}/train/loramcl_s0", exist_ok=True)
    os.makedirs(f"{se}/train/sft1_eq_s0", exist_ok=True)
    for f in (f"{p1}/train/wta1/adapter.pt", f"{se}/train/loramcl_s0/adapter.pt", f"{se}/train/sft1_eq_s0/adapter.pt"):
        open(f, "w").close()
    json.dump({"tokens": 1000}, open(f"{p1}/train/wta1/result.json", "w"))
    r = subprocess.run(prun_cmd(f"{d}/plan", p0, p1, sr, se, "--bench", "math", "--plan_only"), env=ENV,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    P = json.load(open(f"{d}/plan/points.json"))
    names = [p["name"] for p in P["points"]]
    assert names == ["sft", "wta1", "p0", "p2", "p13", "p32", "loramcl"], names
    reused = {p["name"]: p["seeds"]["0"]["new"] for p in P["points"]}
    assert reused == {"sft": False, "wta1": False, "p0": True, "p2": True, "p13": True, "p32": True, "loramcl": False}
    out = r.stdout + r.stderr
    assert "train_p13_s0" in out and "train_p0_s0" in out and "train_sft_s0" not in out and "train_wta1_s0" not in out
    r = subprocess.run(prun_cmd(f"{d}/plan2", p0, p1, sr, se, "--bench", "math", "--plan_only", "--points",
                                "sft,32nf,mcl"), env=ENV, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert [p["name"] for p in json.load(open(f"{d}/plan2/points.json"))["points"]] == ["sft", "p32nf", "mcl"]
    r = subprocess.run(prun_cmd(f"{d}/plan", p0, p1, sr, se, "--bench", "math", "--plan_only", "--points", "sft,40"),
                       env=ENV, capture_output=True, text=True)
    assert r.returncode != 0 and "unknown point" in r.stdout + r.stderr
    r = subprocess.run(prun_cmd(f"{d}/plan_mt", p0, p1, sr, se, "--bench", "mt", "--plan_only"), env=ENV,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    P = json.load(open(f"{d}/plan_mt/points.json"))
    assert [p["name"] for p in P["points"]] == ["sft", "p1", "p6", "p16", "mcl"]
    assert "mt_base" in r.stdout + r.stderr and "eval_mcl_s0" in r.stdout + r.stderr
    # labelled references: kinds, order after the curve, r_p from the number, notes for the report
    r = subprocess.run(prun_cmd(f"{d}/plan_mc", p0, p1, sr, se, "--bench", "math", "--plan_only", "--points",
                                "oracle2,match,sft,sftmc,oracle,wtamc"), env=ENV, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    P = json.load(open(f"{d}/plan_mc/points.json"))["points"]
    assert [p["name"] for p in P] == ["sft", "sftmc", "wtamc", "match", "oracle", "oracle2"], [p["name"] for p in P]
    by = {p["name"]: p for p in P}
    assert by["oracle2"]["r_p"] == 2 and by["sftmc"]["share"] is None and all(p["note"] for p in P)
    # no room for one exported checkpoint under the quota: refused before any job, the largest directories named
    r = subprocess.run(prun_cmd(f"{d}/plan_full", p0, p1, sr, se, "--bench", "math", "--plan_only", "--disk_gb", "3"),
                       env=ENV, capture_output=True, text=True)
    assert r.returncode != 0 and "Free space first" in r.stdout + r.stderr, r.stdout + r.stderr
    for bad in ("sftmc2", "match40"):
        r = subprocess.run(prun_cmd(f"{d}/plan_mc_bad", p0, p1, sr, se, "--bench", "math", "--plan_only", "--points",
                                    bad), env=ENV, capture_output=True, text=True)
        assert r.returncode != 0, bad
    from phase1 import prun
    a = prun.parse(["--bench", "math", "--run", f"{d}/plan_mc", "--phase1", p1, "--seeds_run", se])
    pts = {p["name"]: p for p in prun.parse_points("sftmc,wtamc,match,oracle2", "math", 32)}
    cmd = prun.train_cmd(a, pts["oracle2"], 0, "o", "data", "m")
    assert cmd[cmd.index("--variant") + 1] == "oracle" and cmd[cmd.index("--r_p") + 1] == "2"
    assert cmd[cmd.index("--modc_file") + 1] == f"{os.path.abspath(se)}/modc/train.jsonl"
    cmd = prun.train_cmd(a, pts["wtamc"], 0, "o", "data", "m")
    assert cmd[cmd.index("--variant") + 1] == "wta" and cmd[cmd.index("--film") + 1] == "1" and "--modc_file" in cmd
    cmd = prun.train_cmd(a, pts["sftmc"], 0, "o", "data", "m")
    assert cmd[cmd.index("--variant") + 1] == "sft" and "--modc_file" in cmd
    print("plans: points, reused adapters, refused inputs: ok")
    run = f"{d}/smoke"
    r = subprocess.run(prun_cmd(run, p0, p1, sr, se, "--bench", "math", "--smoke", "--max_tokens", "8"), env=ENV)
    assert r.returncode == 0, "smoke run failed, see " + os.path.join(run, "logs")
    rep = open(f"{run}/report_priv.md", encoding="utf-8").read()
    S = json.load(open(f"{run}/summary_priv.json"))
    assert set(S["rows"]) == {"sft", "p2"} and "p2" in S["delta_vs_sft"] and S["loss"]["val"]["p2"]["win_share"]
    assert "Лосс на эталонах" in rep
    print("math smoke run end to end: ok")
    run = f"{d}/smoke_mc"
    r = subprocess.run(prun_cmd(run, p0, p1, sr, se, "--bench", "math", "--smoke", "--max_tokens", "8", "--points",
                                "sftmc,match,oracle"), env=ENV)
    assert r.returncode == 0, "smoke run failed, see " + os.path.join(run, "logs")
    S = json.load(open(f"{run}/summary_priv.json"))
    assert set(S["rows"]) == {"sft", "sftmc", "wtamc", "match", "oracle"} and "oracle" in S["delta_vs_sft"]
    assert json.load(open(f"{run}/train/oracle_s0/result.json"))["refs"] == "modc"
    print("math smoke run on labelled references end to end: ok")


def main():
    d = tempfile.mkdtemp()
    try:
        base = tiny_model(d)
        test_model(base, d)
        p0, p1, sr, se = fake_phase(d, base)
        test_train(base, d, p1)
        test_train_modc(base, d, p1, se)
        test_refs_nll(base, d, p0, p1, se)
        test_mt(d)
        test_plan_and_smoke(d, base, p0, p1, sr, se)
        print("all ok")
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
