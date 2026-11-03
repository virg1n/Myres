"""Steps of multi-round self-training (phase1/srun.py).

  gen    python -m phase1.selftrain gen --base B --adapter A --data D --out O --sets pool:4,hv:16,test:4 --seed S
         a model solves several sets; --sets gives the attempts per problem of each set (D/<set>.jsonl). A model with
         M modes (FiLM codes or separate adapters) gets attempts / M samples per mode, a model without modes all
         attempts as samples of one checkpoint, the base model (--adapter "") likewise. Outputs:
         solve_<m>_<set>.jsonl.gz (m = mode, 0 without modes), gen.json, DONE. Mode m samples with the vLLM seed
         S + 1000 * (m + 1). FiLM codes are written into a shared export (film.export_shared / write_film_dir, seconds
         per mode), separate adapters are exported one at a time (3.5 GB each). gen.json pins the adapter, the sets
         and the sampling: a restart with others is refused instead of mixing outputs.
  train  the bank of all admitted pool solutions of rounds 0..r (build_bank) -> phase1.train --variant fixed from the
         START adapter (fresh Adam), min(--steps_max, --epochs passes over the bank) steps of --P problems with all
         their solutions; no verifier examples. The frozen set never enters the bank."""
import argparse
import gzip
import json
import math
import os
import shutil
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase0.answers import last_boxed  # noqa: E402
from phase1 import film as FM  # noqa: E402
from phase1.evaluate import CKPT_GB, gen, log, wait_disk  # noqa: E402
from phase1.keval import check_manifest  # noqa: E402
from phase1.train import load_jsonl  # noqa: E402

PY = sys.executable


def parse_sets(spec):
    """'pool:4,hv:16' -> [('pool', 4), ('hv', 16)]"""
    out = []
    for item in spec.split(","):
        name, _, n = item.partition(":")
        out.append((name, int(n)))
    return out


def n_modes(cfg):
    if cfg.get("prefix_modes") or cfg.get("film_z"):
        raise SystemExit("self-training supports FiLM codes, separate adapters or no modes")
    return cfg.get("film_modes", 0) or cfg.get("lora_modes", 0)


def mode_sets(sets, M):
    """Per-mode spec for phase0.gen mode_solve: attempts / M samples per mode (all attempts without modes)."""
    for name, n in sets:
        if M and n % M:
            raise SystemExit(f"{name}: {n} attempts are not a multiple of the {M} modes")
    return ",".join(f"{name}:{n // M if M else n}" for name, n in sets if n)


def done_mode(out, m, sets):
    return all(os.path.exists(os.path.join(out, f"solve_{m}_{name}.jsonl.gz")) for name, n in sets if n)


def read_gz(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(l) for l in f]


def attempts(gdir, name):
    """id -> [(mode or None, text, correct, answer key)] of one generation step on one set (all modes)."""
    man = json.load(open(os.path.join(gdir, "gen.json")))
    M, out = man["modes"], {}
    if name not in man["sets"]:
        return out
    n = man["sets"][name] // M if M else man["sets"][name]
    for m in range(max(M, 1)):
        f = os.path.join(gdir, f"solve_{m}_{name}.jsonl.gz")
        if not os.path.exists(f):
            raise ValueError(f"{f} is missing: {gdir} has {name} for {max(M, 1)} mode(s)")
        for r in read_gz(f):
            if not len(r["texts"]) == len(r["correct"]) == len(r["answers"]) == n:
                raise ValueError(f"{f}: {r['id']} has {len(r['texts'])} attempts, expected {n}")
            out.setdefault(r["id"], []).extend((m if M else None, t, bool(c), k)
                                               for t, c, k in zip(r["texts"], r["correct"], r["answers"]))
    return out


def admitted(text, correct):
    """A solution enters the bank only if its answer is right and it ends with a \\boxed{} answer (no fallback)."""
    return correct and last_boxed(text) is not None


def build_bank(gdirs, probs, kind):
    """Training rows of one round from the pool outputs of generation steps gdirs (rounds 0..r, oldest first).

    Every problem has 4 slots: the modes, or the 4 samples of a round for a model without modes. A slot keeps its
    latest admitted solution. Weights: modes, rftbal and swap 1 / (occupied slots) per solution (each solved problem
    weighs 1); rft 1 per solution (a problem weighs its number of solutions). swap: the bank of a mode model trained
    without modes (mode None)."""
    slots, first = {}, {}                        # id -> {slot: (round, mode, text)}; id -> round of the first admission
    for r, gd in enumerate(gdirs):
        for i, lst in attempts(gd, "pool").items():
            for j, (m, t, c, _) in enumerate(lst):
                if admitted(t, c):
                    s = m if m is not None else j % 4
                    slots.setdefault(i, {})[s] = (r, m, t)
                    first.setdefault(i, r)
    rows = []
    for i in sorted(slots):
        sl = slots[i]
        w = 1.0 if kind == "rft" else 1.0 / len(sl)
        rows.append({"id": i, "question": probs[i]["question"], "stratum": probs[i].get("stratum"), "first": first[i],
                     "sols": [{"solution": t, "mode": None if kind in ("rft", "rftbal", "swap") else m, "w": w,
                               "round": r, "slot": s} for s, (r, m, t) in sorted(sl.items())]})
    return rows


def bank_stats(rows, last):
    """last: index of the newest generation step in the bank."""
    occ = [len(r["sols"]) for r in rows]
    st = {"problems": len(rows), "solutions": sum(occ), "full_share": sum(o == 4 for o in occ) / max(1, len(rows)),
          "by_stratum": {}, "per_slot": [0] * 4, "new_last_round": sum(r["first"] == last for r in rows)}
    for r in rows:
        st["by_stratum"][r["stratum"]] = st["by_stratum"].get(r["stratum"], 0) + 1
        for s in r["sols"]:
            st["per_slot"][s["slot"]] += 1
    return st


def cmd_train(a):
    os.makedirs(a.out, exist_ok=True)
    if os.path.exists(os.path.join(a.out, "DONE")):
        return
    probs = {p["id"]: p for p in load_jsonl(os.path.join(a.data, "pool.jsonl"))}
    gdirs = a.gens.split(",")
    rows = build_bank(gdirs, probs, a.kind)
    bf = os.path.join(a.out, "bank.jsonl")
    with open(bf + ".tmp", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(bf + ".tmp", bf)
    st = bank_stats(rows, len(gdirs) - 1)
    # the same update budget for every method, but at most --epochs passes over a small early bank
    st["steps"] = min(a.steps_max, math.ceil(a.epochs * len(rows) / a.P))
    json.dump(st, open(os.path.join(a.out, "bank.json"), "w"), indent=1)
    log(f"bank ({a.kind}): {st}")
    r = torch.load(a.init, map_location="cpu", weights_only=False)["cfg"]["lora_r"]     # alpha follows the adapter
    # FiLM keeps the LR ratio of phase 1 (film_lr 1e-3 at lr 2e-4); the LR schedule follows the planned steps,
    # --max_hours is only a safety limit far above a normal round
    cmd = [PY, "-m", "phase1.train", "--variant", "fixed", "--base", a.base, "--data", a.data, "--train_file", bf,
           "--init", a.init, "--init_opt", "0", "--lora_r", str(r), "--out", a.out, "--lr", str(a.lr),
           "--film_lr", str(a.lr * a.film_lr_ratio), "--P", str(a.P),
           "--ver_ratio", "0", "--max_hours", str(a.max_hours), "--seed", str(a.seed), "--save_opt", "0",
           "--ckpt_minutes", "1000"]
    if a.kind == "swap":
        cmd += ["--film_modes", "0"]
    if st["steps"]:
        cmd += ["--max_steps", str(st["steps"])]
    if subprocess.call(cmd) != 0:
        raise SystemExit("training failed")


def cmd_gen(a):
    os.makedirs(a.out, exist_ok=True)
    sets = parse_sets(a.sets)
    ad = os.path.abspath(a.adapter) if a.adapter else ""
    if ad and not os.path.exists(ad) and os.path.exists(os.path.join(a.out, "DONE")):
        return                           # finished, and the intermediate adapter was dropped
    check_manifest(a.out, {"adapter": ad, "adapter_bytes": os.path.getsize(ad) if ad else 0,
                           "adapter_mtime": int(os.path.getmtime(ad)) if ad else 0, "base": os.path.abspath(a.base),
                           "data": os.path.abspath(a.data), "sets": a.sets, "seed": a.seed,
                           "max_tokens": a.max_tokens})
    if os.path.exists(os.path.join(a.out, "DONE")):
        return
    ck = torch.load(ad, map_location="cpu", weights_only=False) if ad else None
    cfg = ck["cfg"] if ck else {}
    M = n_modes(cfg)
    spec = mode_sets(sets, M)
    todo = [m for m in range(max(M, 1)) if not done_mode(a.out, m, sets)]
    tmp = a.tmp or os.path.join(a.out, "tmp_export")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    try:
        if todo:
            film, shared = None, None
            if cfg.get("film_modes"):
                from transformers import AutoConfig
                mc = AutoConfig.from_pretrained(a.base)
                film = FM.load_film(cfg, ck["adapter"], 2 * mc.num_hidden_layers + 1, mc.hidden_size)
                wait_disk(tmp, CKPT_GB + 2)
                shared = FM.export_shared(a.base, ad, os.path.join(tmp, "shared"))
            d = os.path.join(tmp, "model")       # one path for every mode: vLLM keys its compile cache by the path
            for m in todo:
                t0 = time.time()
                if film is not None:
                    with torch.no_grad():
                        FM.write_film_dir(shared, d, film.gamma(torch.tensor([m]))[0])
                    model = d
                elif ad:
                    wait_disk(tmp, CKPT_GB + 2)
                    FM.export_modes(a.base, ad, [d], [m if M else None])
                    model = d
                else:
                    model = a.base
                gen("mode_solve", model, a.data, a.out, ["--tag", str(m), "--sets", spec,
                                                         "--seed", str(a.seed + 1000 * (m + 1))],
                    max_tokens=a.max_tokens)
                shutil.rmtree(d, ignore_errors=True)
                log(f"mode {m + 1}/{max(M, 1)} ({spec}) in {(time.time() - t0) / 60:.1f} min")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    json.dump({"modes": M, "sets": dict(sets), "seed": a.seed, "adapter": ad}, open(os.path.join(a.out, "gen.json"), "w"))
    open(os.path.join(a.out, "DONE"), "w").close()
    if a.drop_adapter and ad:            # an intermediate model is needed only for its own generation
        os.remove(ad)
    log("generation done", a.out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--base", required=True)
    g.add_argument("--adapter", default="", help="empty: the base model")
    g.add_argument("--data", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--tmp", default="")
    g.add_argument("--sets", required=True, help="name:attempts,...")
    g.add_argument("--seed", type=int, required=True)
    g.add_argument("--max_tokens", type=int, default=1200)
    g.add_argument("--drop_adapter", type=int, default=0, help="1: delete --adapter once the outputs are written")
    t = sub.add_parser("train")
    t.add_argument("--base", required=True)
    t.add_argument("--init", required=True, help="the start adapter: every round restarts from it")
    t.add_argument("--gens", required=True, help="comma-separated generation directories, rounds 0..r")
    t.add_argument("--kind", required=True, choices=["modes", "rft", "rftbal", "swap"])
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--steps_max", type=int, default=300)
    t.add_argument("--epochs", type=float, default=2.0, help="at most this many passes over the bank")
    t.add_argument("--P", type=int, default=16, help="problems per step (all their bank solutions)")
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--film_lr_ratio", type=float, default=5.0, help="FiLM LR / LoRA LR (phase 1: 1e-3 / 2e-4)")
    t.add_argument("--max_hours", type=float, default=2.0, help="safety limit of one round (the report flags it)")
    t.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"gen": cmd_gen, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
