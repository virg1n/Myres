"""The en->de translation benchmark of LoRA-MCL (arXiv 2507.10419, github.com/Victorletzelter/LoRA-MCL, ALMA/) for the
private-share sweep (phase1/prun.py --bench mt).

  data   train: ALMA/human_written_data/deen/train.de-en.{en,de} of the LoRA-MCL repo (the human-written pairs of
         ALMA, ~14k); test: newstest2014 with 10 extra references per sentence (Ott et al. 2018,
         dl.fbaipublicfiles.com/fairseq/data/wmt14-en-de.extra_refs.tok, lines S-/T-/R1..R10-), the file their
         evaluation script reads; as there, the tokenized source (S line) is the input as it is
  base   haoranxu/ALMA-7B-Pretrain (their base) is 27 GB of float32 .bin shards: one shard at a time is downloaded to
         --tmp (default /dev/shm: RAM, not the disk quota), converted to bfloat16 safetensors in --out and deleted,
         so the disk holds 13.5 GB at the end and never more
  eval   one model (sweep point), in process with Hugging Face generate (no export, no vLLM):
           hypotheses: modes: one per mode, beam search of width b (--beams, the repo decodes with 1, 2, K beams),
             do_sample False, 256 new tokens; without modes: the K best beams of a beam search of width b * K
             (the LoRA-MLE beam baseline of the paper); the output is cleaned as ALMA does (first non-empty line)
           metrics, ported from their compute_metrics.py (hypotheses and references Moses-detokenized, lang de):
             corpus BLEU of every hypothesis against the first reference (13a, their corpus_bleu_scores) and against
             all 10, oracle sentence BLEU (mean over sentences of the best hypothesis, all references), Pairwise-BLEU
             and leave-one-out multi-reference BLEU of fairseq (tokenize none and 13a), references covered
           loss: NLL of every mode on every (detokenized) reference, summed over its tokens (best mode vs SFT;
             which mode wins which reference: the 10 references of one sentence split over the modes or not)
         outputs hyps_b<b>.json, metrics.json, nll.json.gz, DONE"""
import argparse
import gzip
import json
import os
import shutil
import sys
import time
import urllib.request

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase1 import priv as PV  # noqa: E402
from phase1.ptrain import encode_mt, mt_prompt  # noqa: E402
from phase1.train import collate, load_jsonl, micro, per_example_nll  # noqa: E402

REPO = "https://raw.githubusercontent.com/Victorletzelter/LoRA-MCL/main/ALMA/human_written_data/deen/"
EXTRA_REFS = "https://dl.fbaipublicfiles.com/fairseq/data/wmt14-en-de.extra_refs.tok"
BASE_REPO = "haoranxu/ALMA-7B-Pretrain"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def write_jsonl(path, rows):
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(path + ".tmp", path)


def fetch(src, dst):
    """A local path or a URL -> dst."""
    if os.path.exists(src):
        shutil.copy(src, dst)
        return
    req = urllib.request.Request(src, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dst + ".tmp", "wb") as f:
        shutil.copyfileobj(r, f)
    os.replace(dst + ".tmp", dst)


def load_ref(path):
    """fairseq examples/translation_moe/score.py: S-<i>, T-<i> and a block of R<n>-<i> lines per sentence."""
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    src, tgt, refs = [], [], []
    i = 0
    while i < len(lines):
        if lines[i].startswith("S-"):
            src.append(lines[i].split("\t")[1].rstrip())
            i += 1
        elif lines[i].startswith("T-"):
            tgt.append(lines[i].split("\t")[1].rstrip())
            i += 1
        elif lines[i].startswith("R"):
            a = []
            while i < len(lines) and lines[i].startswith("R"):
                a.append(lines[i].split("\t")[1].rstrip())
                i += 1
            refs.append(a)
        else:
            i += 1
    return src, tgt, refs


# ----------------------------------------------------------------------------- data
def cmd_data(a):
    os.makedirs(a.out, exist_ok=True)
    if os.path.exists(os.path.join(a.out, "DONE")):
        return
    raw = os.path.join(a.out, "raw")
    os.makedirs(raw, exist_ok=True)
    for src, name in ((a.train_en, "train.en"), (a.train_de, "train.de"), (a.test_file, "test.extra_refs.tok")):
        if not os.path.exists(os.path.join(raw, name)):
            log("fetch", src)
            fetch(src, os.path.join(raw, name))
    en = open(os.path.join(raw, "train.en"), encoding="utf-8").read().split("\n")
    de = open(os.path.join(raw, "train.de"), encoding="utf-8").read().split("\n")
    while en and not en[-1] and de and not de[-1]:
        en.pop()
        de.pop()
    if len(en) != len(de):
        raise SystemExit(f"train: {len(en)} English and {len(de)} German lines")
    train = [{"src": s.strip(), "tgt": t.strip()} for s, t in zip(en, de) if s.strip() and t.strip()]
    src, tgt, refs = load_ref(os.path.join(raw, "test.extra_refs.tok"))
    if not (len(src) == len(tgt) == len(refs)) or not src:
        raise SystemExit(f"test: {len(src)} sources, {len(tgt)} targets, {len(refs)} reference blocks")
    nref = sorted({len(r) for r in refs})
    if nref[0] < 2:
        raise SystemExit(f"test: sentences with {nref[0]} reference(s); the extra references are missing")
    if a.max_train:
        train = train[:a.max_train]
    test = [{"i": i, "src": s, "tgt": t, "refs": r} for i, (s, t, r) in enumerate(zip(src, tgt, refs))]
    if a.max_test:
        test = test[:a.max_test]
    write_jsonl(os.path.join(a.out, "train.jsonl"), train)
    write_jsonl(os.path.join(a.out, "test.jsonl"), test)
    st = {"train": len(train), "dropped_empty": len(en) - len(train), "test": len(test), "refs_per_sentence": nref}
    json.dump(st, open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    if not a.max_train and not (10_000 <= len(train) <= 20_000):
        log(f"WARNING: {len(train)} training pairs, the paper says ~14k")
    if not a.max_test and len(test) != 500:
        log(f"WARNING: {len(test)} test sentences, expected 500")
    open(os.path.join(a.out, "DONE"), "w").close()
    log("mt data:", st)


# ----------------------------------------------------------------------------- base model
def default_tmp(need_gb):
    if os.path.isdir("/dev/shm") and shutil.disk_usage("/dev/shm").free / 1e9 > need_gb:
        return "/dev/shm"
    return ""


def cmd_base(a):
    """haoranxu/ALMA-7B-Pretrain -> bfloat16 safetensors, one shard at a time."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import save_file
    os.makedirs(a.out, exist_ok=True)
    if os.path.exists(os.path.join(a.out, "DONE")):
        return
    parent = a.tmp or default_tmp(11)
    if not parent:
        raise SystemExit("no --tmp with 11 GB free outside the disk quota (/dev/shm too small or missing): pass --tmp "
                         "(a .bin shard of 9.9 GB is held there while it is converted)")
    tmp = os.path.join(parent, "alma_shards")        # our own directory: only it is deleted, never --tmp itself
    os.makedirs(tmp, exist_ok=True)
    small = ["config.json", "generation_config.json", "tokenizer.model", "tokenizer_config.json",
             "special_tokens_map.json", "pytorch_model.bin.index.json"]
    for f in small:
        try:
            hf_hub_download(a.repo, f, local_dir=tmp)
        except Exception as e:
            if f in ("config.json", "tokenizer.model", "pytorch_model.bin.index.json"):
                raise
            log(f"{f}: not downloaded ({e!r:.100})")
    idx = json.load(open(os.path.join(tmp, "pytorch_model.bin.index.json")))
    shards = list(dict.fromkeys(idx["weight_map"].values()))
    wmap, total = {}, 0
    for i, sh in enumerate(shards):
        name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        dst = os.path.join(a.out, name)
        if not os.path.exists(dst):
            t0 = time.time()
            p = hf_hub_download(a.repo, sh, local_dir=tmp)
            try:
                sd = torch.load(p, map_location="cpu", weights_only=True, mmap=True)
            except RuntimeError:
                sd = torch.load(p, map_location="cpu", weights_only=True)
            out = {k: v.to(torch.bfloat16).contiguous() for k, v in sd.items()}
            del sd
            save_file(out, dst + ".tmp", metadata={"format": "pt"})
            os.replace(dst + ".tmp", dst)
            os.remove(p)
            log(f"{sh} -> {name} ({len(out)} tensors) in {(time.time() - t0) / 60:.1f} min")
            del out
        from safetensors import safe_open
        with safe_open(dst, "pt") as f:
            for k in f.keys():
                wmap[k] = name
                t = f.get_slice(k)
                total += int(np.prod(t.get_shape())) * 2
    json.dump({"metadata": {"total_size": total}, "weight_map": wmap},
              open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=1)
    cfg = json.load(open(os.path.join(tmp, "config.json")))
    cfg["torch_dtype"] = "bfloat16"
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    for f in small[1:-1]:
        if os.path.exists(os.path.join(tmp, f)):
            shutil.copy(os.path.join(tmp, f), os.path.join(a.out, f))
    shutil.rmtree(tmp, ignore_errors=True)
    open(os.path.join(a.out, "DONE"), "w").close()
    log("base model ready:", a.out)


# ----------------------------------------------------------------------------- metrics (their compute_metrics.py)
def detok(lines):
    from sacremoses import MosesDetokenizer
    md = MosesDetokenizer(lang="de")
    return [md.detokenize(x.split()) for x in lines]


def clean(text):
    """ALMA's clean_outputstring after the prompt: the first non-empty line (of the first three)."""
    out = text.split("\n")
    for x in out[:3]:
        if x.strip():
            return x.strip()
    return ""


def corpus_bleu(sys_stream, ref_streams, tokenize="none"):
    import sacrebleu
    return sacrebleu.corpus_bleu(sys_stream, ref_streams, tokenize=tokenize).score


def pairwise_bleu(hyps, tokenize="none"):
    """fairseq Pairwise-BLEU: every hypothesis against every other one of its sentence (pairwise_bleu_fairseq_exact)."""
    K, N = len(hyps), len(hyps[0])
    if K < 2:
        return float("nan")
    refs_flat, hyps_flat = [], []
    for n in range(N):
        S = [hyps[k][n] for k in range(K)]
        for i in range(K):
            for j in range(K):
                if i != j:
                    refs_flat.append(S[i])
                    hyps_flat.append(S[j])
    return corpus_bleu(hyps_flat, [refs_flat], tokenize)


def loo_bleu(refs, hyps, tokenize="none"):
    """fairseq leave-one-out multi-reference BLEU (leave_one_out_multi_ref_bleu)."""
    N, K = len(refs), len(hyps)
    if all(len(r) == 1 for r in refs):
        return float("nan")
    M = min(len(r) for r in refs)
    by_index = list(zip(*[r[:M] for r in refs]))
    flat = [hyps[k][i] for i in range(N) for k in range(K)]
    dup = [[s for s in stream for _ in range(K)] for stream in by_index]
    return float(np.mean([corpus_bleu(flat, dup[:h] + dup[h + 1:], tokenize) for h in range(M)]))


def refs_covered(refs, hyps, tokenize="13a"):
    from sacrebleu import sentence_bleu
    tot = 0
    for i in range(len(refs)):
        chosen = set()
        for k in range(len(hyps)):
            s = [sentence_bleu(hyps[k][i], [r], tokenize=tokenize).score for r in refs[i]]
            chosen.add(int(np.argmax(s)))
        tot += len(chosen)
    return tot / len(refs)


def oracle_sentence_bleu(refs, hyps, tokenize="13a"):
    from sacrebleu import sentence_bleu
    sc = np.array([[sentence_bleu(hyps[k][i], refs[i], tokenize=tokenize).score for i in range(len(refs))]
                   for k in range(len(hyps))])
    return float(sc.max(0).mean()), sc


def metrics(hyps, refs_tok):
    """hyps [K][N] (cleaned outputs), refs_tok [N][M] (tokenized references of the test file)."""
    H = [detok(h) for h in hyps]
    R = [detok(r) for r in refs_tok]
    first = [r[0] for r in R]
    M = min(len(r) for r in R)
    multi = [list(x) for x in zip(*[r[:M] for r in R])]
    oracle, sc = oracle_sentence_bleu(R, H)
    out = {"K": len(H), "N": len(R),
           "bleu_first_ref": [corpus_bleu(h, [first], "13a") for h in H],
           "bleu_all_refs": [corpus_bleu(h, multi, "13a") for h in H],
           "oracle_sentence_bleu": oracle,
           "sentence_bleu_mean": float(sc.mean()),
           "pairwise_bleu": pairwise_bleu(H, "none"), "pairwise_bleu_13a": pairwise_bleu(H, "13a"),
           "loo_bleu": loo_bleu(R, H, "none"), "loo_bleu_13a": loo_bleu(R, H, "13a"),
           "refs_covered": refs_covered(R, H)}
    out["bleu_first_ref_mean"] = float(np.mean(out["bleu_first_ref"]))
    out["bleu_all_refs_mean"] = float(np.mean(out["bleu_all_refs"]))
    out["oracle_per_sentence"] = sc.max(0).tolist()
    out["sentence_bleu"] = sc.tolist()
    return out


# ----------------------------------------------------------------------------- evaluation of one model
@torch.no_grad()
def generate(model, tok, prompts, beams, ret, bs, max_new, dev):
    out = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=True,
                  return_token_type_ids=False).to(dev)
        g = model.generate(**enc, max_new_tokens=max_new, num_beams=beams, num_return_sequences=ret, do_sample=False,
                           use_cache=True, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                           temperature=None, top_p=None, top_k=None)
        texts = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for r, i in enumerate(idx):
            out[i] = [clean(t) for t in texts[r * ret:(r + 1) * ret]]
    return out


def cmd_eval(a):
    from phase1.keval import check_manifest
    os.makedirs(a.out, exist_ok=True)
    tf = os.path.join(a.data, "test.jsonl")
    check_manifest(a.out, {"adapter": os.path.abspath(a.adapter), "adapter_bytes": os.path.getsize(a.adapter),
                           "adapter_mtime": int(os.path.getmtime(a.adapter)), "base": os.path.abspath(a.base),
                           "test": os.path.abspath(tf), "test_bytes": os.path.getsize(tf), "beams": a.beams,
                           "K": a.K, "max_new": a.max_new, "nll": a.nll})
    if os.path.exists(os.path.join(a.out, "DONE")):
        return
    with torch.no_grad():
        evaluate(a)
    open(os.path.join(a.out, "DONE"), "w").close()
    log("mt eval done", a.out)


def evaluate(a):
    from transformers import AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.base)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.unk_token or tok.eos_token
    model, film, ctx, c = PV.build_from(a.base, PV.load_ckpt(a.adapter), dev)
    model.eval()
    modes = c["r_p"] > 0 or c["film"]
    K = c["K"] if modes else a.K
    test = load_jsonl(os.path.join(a.data, "test.jsonl"))
    prompts = [mt_prompt(r["src"]) for r in test]
    refs = [r["refs"] for r in test]
    res = {"cfg": c, "modes": bool(modes), "K": K, "beams": {}}
    for b in a.beams:
        f = os.path.join(a.out, f"hyps_b{b}.json")
        if os.path.exists(f):
            hyps = json.load(open(f))
        else:
            t0 = time.time()
            if modes:
                hyps = []
                for m in range(K):
                    PV.set_mode(ctx, film, m, device=dev)
                    hyps.append([h[0] for h in generate(model, tok, prompts, b, 1, a.bs, a.max_new, dev)])
            else:
                PV.set_mode(ctx, film, None)
                g = generate(model, tok, prompts, b * K, K, max(1, a.bs // K), a.max_new, dev)
                hyps = [[g[i][k] for i in range(len(g))] for k in range(K)]
            json.dump(hyps, open(f + ".tmp", "w", encoding="utf-8"), ensure_ascii=False)
            os.replace(f + ".tmp", f)
            log(f"beam {b}: {K} hypotheses of {len(prompts)} sentences in {(time.time() - t0) / 60:.1f} min")
        res["beams"][str(b)] = metrics(hyps, refs)
        m = res["beams"][str(b)]
        log(f"beam {b}: BLEU {m['bleu_first_ref_mean']:.2f}, oracle {m['oracle_sentence_bleu']:.2f}, "
            f"pairwise {m['pairwise_bleu']:.2f}, loo {m['loo_bleu']:.2f}")
    if a.nll:
        ex, rid = [], []
        for r in test:
            for j, ref in enumerate(detok(r["refs"])):
                p, t, _ = encode_mt(tok, r["src"], ref)
                ex.append(dict(p=p, t=t))
                rid.append((r["i"], j))
        Km = K if modes else 1
        nll = torch.zeros(Km, len(ex), dtype=torch.float64)
        pad = tok.pad_token_id
        for k in range(Km):
            md = torch.tensor([k], device=dev) if modes else None
            for ids in micro(ex, a.mb_tokens):
                x, att, lab = collate(ex, ids, pad, dev)
                with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                    v = per_example_nll(model, film, ctx, x, att, lab, md).double().cpu()
                nll[k, ids] = v * torch.tensor([len(ex[i]["t"]) for i in ids], dtype=torch.float64)
        with gzip.open(os.path.join(a.out, "nll.json.gz"), "wt") as g:
            json.dump({"modes": Km, "ref": rid, "ntok": [len(e["t"]) for e in ex], "nll": nll.tolist()}, g)
    json.dump(res, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("--out", required=True)
    d.add_argument("--train_en", default=REPO + "train.de-en.en")
    d.add_argument("--train_de", default=REPO + "train.de-en.de")
    d.add_argument("--test_file", default=EXTRA_REFS)
    d.add_argument("--max_train", type=int, default=0)
    d.add_argument("--max_test", type=int, default=0)
    b = sub.add_parser("base")
    b.add_argument("--repo", default=BASE_REPO)
    b.add_argument("--out", required=True)
    b.add_argument("--tmp", default="", help="for one .bin shard (9.9 GB); default /dev/shm")
    e = sub.add_parser("eval")
    e.add_argument("--base", required=True)
    e.add_argument("--adapter", required=True)
    e.add_argument("--data", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--beams", default="1,3")
    e.add_argument("--K", type=int, default=3, help="hypotheses of a model without modes (its best beams)")
    e.add_argument("--bs", type=int, default=16)
    e.add_argument("--max_new", type=int, default=256)
    e.add_argument("--nll", type=int, default=1)
    e.add_argument("--mb_tokens", type=int, default=8192)
    a = ap.parse_args(argv)
    if a.cmd == "eval":
        a.beams = [int(x) for x in a.beams.split(",")]
    {"data": cmd_data, "base": cmd_base, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
