"""Evaluation of the private-share sweep on math (phase1/prun.py --bench math).

  refs   held-out references with their strategy (no generation): every problem of phase-1 val (never trained on)
         gets the first correct solution of each of the 4 phase-0 strategies (system prompts) from the pool outputs;
         --n_train problems of the ModC training data (their references were trained on in part) the same way
  nll    loss of every mode on every reference (neutral prompt, mean over target tokens, as in training): the loss
         of the best mode against SFT (Table 1 of LoRA-MCL) and which mode wins which reference (by strategy, source,
         length, problem)
  gen    a set (hv) solved by every mode: k / K samples per mode at the sampling of phase 0 (T 0.7, top-p 0.8,
         top-k 20) + one greedy solution per mode; a model without modes: k samples + one greedy. Mode m samples
         with the vLLM seed --seed + 1000 * (m + 1). Every mode is exported to a plain checkpoint (3.5 GB) one at a
         time and deleted after its generation. Outputs solve_<m>_test.jsonl.gz, greedy_<m>_test.jsonl.gz (the set
         is copied to <out>/set/test.jsonl for phase0.gen k_solve), peval.json, DONE; keval.json pins the inputs."""
import argparse
import gzip
import json
import os
import random
import shutil
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase1 import priv as PV  # noqa: E402
from phase1.evaluate import CKPT_GB, gen, log, wait_disk  # noqa: E402
from phase1.keval import check_manifest  # noqa: E402
from phase1.train import collate, encode_solve, load_jsonl, micro, per_example_nll  # noqa: E402


def read_gz(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(l) for l in f]


def write_gz(path, rows):
    with gzip.open(path + ".tmp", "wt") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(path + ".tmp", path)


def ref_key(r):
    """Identity of a reference: the report pairs losses with references by it, not by position alone."""
    return f"{r['split']}|{r['id']}|{r['strategy']}"


def first_correct(pool_dirs, ids):
    """id -> {strategy: first correct text} from the phase-0 / hard pool outputs."""
    per = {}
    for d in pool_dirs:
        for name in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if name.startswith("pool_") and name.endswith(".jsonl.gz"):
                for r in read_gz(os.path.join(d, name)):
                    if r["id"] in ids and isinstance(r.get("mode"), int):
                        for t, c in zip(r["texts"], r["correct"]):
                            if c and t:
                                per.setdefault(r["id"], {}).setdefault(r["mode"], t)
    return per


def cmd_refs(a):
    val = load_jsonl(os.path.join(a.phase1, "data", "val.jsonl"))
    per = first_correct([os.path.join(a.phase0, "pool"), os.path.join(a.phase1, "hard", "out")], {p["id"] for p in val})
    rows = []
    for p in val:
        for k, t in sorted(per.get(p["id"], {}).items()):
            rows.append({"id": p["id"], "split": "val", "source": str(p.get("source")), "question": p["question"],
                         "strategy": k, "text": t})
    n_val = len({r["id"] for r in rows})
    if a.modc and os.path.exists(a.modc) and a.n_train:
        tr = load_jsonl(a.modc)
        random.Random(0).shuffle(tr)
        for p in tr[:a.n_train]:
            for ref in p["refs"]:
                rows.append({"id": p["id"], "split": "train", "source": str(p.get("source")), "question": p["question"],
                             "strategy": ref["mode"], "text": ref["text"]})
    elif a.n_train:
        log(f"no ModC data at {a.modc}: held-out references only")
    if n_val < 50:
        raise SystemExit(f"only {n_val} val problems with a correct solution in the pools")
    os.makedirs(a.out, exist_ok=True)
    write_gz(os.path.join(a.out, "refs.jsonl.gz"), rows)
    open(os.path.join(a.out, "DONE"), "w").close()
    log(f"references: {sum(r['split'] == 'val' for r in rows)} held-out on {n_val} problems, "
        f"{sum(r['split'] == 'train' for r in rows)} from training")


@torch.no_grad()
def cmd_nll(a):
    """<out>/nll.json.gz: nll [modes][refs] summed over target tokens, ntok [refs]."""
    os.makedirs(a.out, exist_ok=True)
    check_manifest(a.out, {"adapter": os.path.abspath(a.adapter), "adapter_bytes": os.path.getsize(a.adapter),
                           "adapter_mtime": int(os.path.getmtime(a.adapter)), "base": os.path.abspath(a.base),
                           "refs": os.path.abspath(a.refs), "refs_bytes": os.path.getsize(a.refs),
                           "refs_mtime": int(os.path.getmtime(a.refs))})
    if os.path.exists(os.path.join(a.out, "DONE")):
        return
    from transformers import AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.base)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model, film, ctx, c = PV.build_from(a.base, PV.load_ckpt(a.adapter), dev)
    model.eval()
    rows = read_gz(a.refs)
    ex = []
    for r in rows:
        p, t = encode_solve(tok, r["question"], r["text"])
        ex.append(dict(p=p, t=t))
    K = c["K"] if (c["r_p"] or c["film"]) else 1
    nll = torch.zeros(K, len(ex), dtype=torch.float64)
    t0 = time.time()
    for k in range(K):
        md = torch.tensor([k], device=dev) if K > 1 else None
        for ids in micro(ex, a.mb_tokens):
            x, att, lab = collate(ex, ids, pad, dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                v = per_example_nll(model, film, ctx, x, att, lab, md).double().cpu()
            nll[k, ids] = v * torch.tensor([len(ex[i]["t"]) for i in ids], dtype=torch.float64)
    out = {"modes": K, "ntok": [len(e["t"]) for e in ex], "nll": nll.tolist(), "keys": [ref_key(r) for r in rows], "refs": os.path.abspath(a.refs),
           "n_refs": len(rows), "adapter": os.path.abspath(a.adapter)}
    f = os.path.join(a.out, "nll.json.gz")
    with gzip.open(f + ".tmp", "wt") as g:
        json.dump(out, g)
    os.replace(f + ".tmp", f)
    open(os.path.join(a.out, "DONE"), "w").close()
    log(f"NLL of {K} mode(s) on {len(ex)} references in {(time.time() - t0) / 60:.1f} min")


def disk_used_gb(root):
    total = 0
    for d, _, files in os.walk(root):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(d, f))
            except OSError:
                pass
    return total / 1e9


class export_slot:
    """One export at a time under --disk_root, started only when what is on that tree plus one checkpoint fits in
    --disk_gb (a quota smaller than the free space the OS reports); the lock serializes the check and the write, so two
    jobs never both see the same room. Without fcntl (Windows) only the size check."""

    def __init__(self, root, limit_gb, need_gb, timeout_h=12.0):     # other evaluations free their export within hours
        self.root, self.limit, self.need, self.timeout = root, limit_gb, need_gb, timeout_h

    def __enter__(self):
        self.f = None
        if not self.root:
            return self
        os.makedirs(self.root, exist_ok=True)
        try:
            import fcntl
            self.f = open(os.path.join(self.root, ".export.lock"), "w")
            fcntl.flock(self.f, fcntl.LOCK_EX)
        except ImportError:
            pass
        t0 = time.time()
        while disk_used_gb(self.root) + self.need > self.limit:
            if time.time() - t0 > self.timeout * 3600:
                raise SystemExit(f"{self.root}: no room for {self.need} GB under {self.limit} GB for {self.timeout} h")
            log(f"waiting: {disk_used_gb(self.root):.1f} GB under {self.root} + {self.need} GB > {self.limit} GB")
            time.sleep(60)
        return self

    def __exit__(self, *exc):
        if self.f is not None:
            self.f.close()


def cmd_gen(a):
    os.makedirs(a.out, exist_ok=True)
    ck = PV.load_ckpt(a.adapter)
    c = ck["cfg"]
    K = c["K"] if (c["r_p"] or c["film"]) else 1
    if a.k % K:
        raise SystemExit(f"k = {a.k} is not a multiple of the {K} modes")
    check_manifest(a.out, {"adapter": os.path.abspath(a.adapter), "adapter_bytes": os.path.getsize(a.adapter),
                           "adapter_mtime": int(os.path.getmtime(a.adapter)), "base": os.path.abspath(a.base),
                           "set": os.path.abspath(a.set_file), "set_bytes": os.path.getsize(a.set_file), "k": a.k,
                           "greedy": a.greedy, "seed": a.seed, "max_tokens": a.max_tokens})
    if os.path.exists(os.path.join(a.out, "DONE")):
        return
    sd = os.path.join(a.out, "set")
    os.makedirs(sd, exist_ok=True)
    shutil.copy(a.set_file, os.path.join(sd, "test.jsonl"))
    tmp = os.path.join(a.tmp or a.out, f"tmp_export_{os.path.basename(os.path.abspath(a.out))}")   # ours only
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    d = os.path.join(tmp, "model")           # one path for every mode: vLLM keys its compile cache by the path
    try:
        for m in range(K):
            if os.path.exists(os.path.join(a.out, f"solve_{m}_test.jsonl.gz")) and \
                    (not a.greedy or os.path.exists(os.path.join(a.out, f"greedy_{m}_test.jsonl.gz"))):
                continue
            t0 = time.time()
            wait_disk(tmp, CKPT_GB + 2)
            with export_slot(a.disk_root, a.disk_gb, CKPT_GB + 0.5):
                PV.export_mode(a.base, ck, d, m if K > 1 else None)
            gen("k_solve", d, sd, a.out, ["--tag", str(m), "--n", str(a.k // K), "--greedy", str(a.greedy),
                                          "--seed", str(a.seed + 1000 * (m + 1))], max_tokens=a.max_tokens)
            shutil.rmtree(d, ignore_errors=True)
            log(f"mode {m + 1}/{K} in {(time.time() - t0) / 60:.1f} min")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    json.dump({"modes": K, "k": a.k, "greedy": a.greedy, "seed": a.seed, "max_tokens": a.max_tokens,
               "cfg": c}, open(os.path.join(a.out, "peval.json"), "w"), indent=1)
    open(os.path.join(a.out, "DONE"), "w").close()
    log("generation done", a.out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("refs")
    r.add_argument("--phase0", required=True)
    r.add_argument("--phase1", required=True)
    r.add_argument("--modc", default="", help="ModC training data (train.jsonl with refs labelled by strategy)")
    r.add_argument("--n_train", type=int, default=600)
    r.add_argument("--out", required=True, help="directory: refs.jsonl.gz, DONE")
    n = sub.add_parser("nll")
    n.add_argument("--base", required=True)
    n.add_argument("--adapter", required=True)
    n.add_argument("--refs", required=True, help="refs.jsonl.gz of the refs step")
    n.add_argument("--out", required=True)
    n.add_argument("--mb_tokens", type=int, default=8192)
    g = sub.add_parser("gen")
    g.add_argument("--base", required=True)
    g.add_argument("--adapter", required=True)
    g.add_argument("--set_file", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--tmp", default="")
    g.add_argument("--k", type=int, default=16)
    g.add_argument("--greedy", type=int, default=1)
    g.add_argument("--seed", type=int, default=17)
    g.add_argument("--max_tokens", type=int, default=1200)
    g.add_argument("--disk_root", default="", help="the tree the disk quota is measured on (e.g. runs/)")
    g.add_argument("--disk_gb", type=float, default=24.0, help="quota of --disk_root")
    a = ap.parse_args(argv)
    {"refs": cmd_refs, "nll": cmd_nll, "gen": cmd_gen}[a.cmd](a)


if __name__ == "__main__":
    main()
