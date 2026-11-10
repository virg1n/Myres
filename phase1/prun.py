"""Private-share sweep with one command: shared LoRA + a private LoRA per mode (phase1/priv.py), FiLM codes in every
point with modes, against SFT and LoRA-MCL, on two benchmarks.

Per-mode LoRA rank R = r_s + r_p is fixed (math 32 as phase 1, mt 16 as LoRA-MCL), so every point has the same scale
alpha / R and the same cost per example; the private share r_p / R goes from 0 (wta1: everything shared) to 1 (K
separate adapters as LoRA-MCL, plus FiLM). A point of --points is r_p, or a keyword.

  --bench math   phase-1 data and the recipe of wta1 (P 16, LR 2e-4, FiLM 1e-3, verifier examples), every point up to
                 the training tokens of wta1 of the same seed (the friend's control: equal tokens, not equal hours)
                 evaluation on hv of phase1_self (held-out hard problems): 16 attempts (4 per mode) + one greedy
                 solution per mode; loss of every mode on held-out references labelled by strategy (peval.py)
                 keywords: sft   the references of wta1 without modes, the same tokens (sft1_eq of phase1_seeds when
                                 it exists, else trained here)
                           0     wta1 itself (phase 1 / phase1_seeds; trained here if missing)
                           loramcl  LoRA-MCL of phase1_seeds (4 x r 32, no FiLM, 2 h by time: ~half the tokens),
                                 evaluated only, a reference outside the curve
                 default sft,0,2,13,32,loramcl: three new trainings per seed (6%, 41%, 100% private)
  --bench mt     the en->de benchmark of LoRA-MCL (phase1/pmt.py): ALMA-7B-Pretrain, ~14k pairs, 2 epochs, LR 2e-5,
                 256 pairs per step, K = 3, R = 16, LoRA dropout 0.05
                 keywords: sft (LoRA-MLE r 16), mcl (LoRA-MCL as in the paper: 3 x r 16, relaxed WTA eps 0.05, no
                 FiLM; outside the curve)
                 default sft,mcl,1,6,16 (6%, 38%, 100% private)
  flow   math: refs -> train (new points) -> gen + nll -> report; mt: data + base -> train -> eval -> report

Disk (25 GB quota): math peaks at ~18 GB (one 3.5 GB export per running evaluation, last.pt with Adam moments of the
running trainings); mt holds the 13.5 GB base (converted shard by shard through /dev/shm) and small adapters. Run the
two benchmarks one after the other, never at the same time."""
import argparse
import json
import os
import shutil
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase1.seeds_run import PY, eval_seed, log  # noqa: E402
from pilot.common import is_done  # noqa: E402
from pilot.run_all import Job, Runner, simulate  # noqa: E402

DEFAULTS = {"math": "sft,0,2,13,32,wta1,loramcl", "mt": "sft,mcl,1,6,16"}
RANK = {"math": 32, "mt": 16}
MODES = {"math": 4, "mt": 3}


def wta1_dir(a, s):
    return f"{a.phase1}/train/wta1" if s == 0 else f"{a.seeds_run}/train/wta1_s{s}"


def parse_points(spec, bench, R):
    pts = []
    for x in [t.strip() for t in spec.split(",") if t.strip()]:
        if x in ("sft",):
            pts.append({"name": "sft", "kind": "sft", "r_p": 0, "share": None, "curve": False})
        elif x == "loramcl" and bench == "math":
            pts.append({"name": "loramcl", "kind": "existing_mcl", "r_p": R, "share": 1.0, "curve": False})
        elif x == "wta1" and bench == "math":
            pts.append({"name": "wta1", "kind": "existing_wta1", "r_p": 0, "share": 0.0, "curve": False})
        elif x == "mcl":
            pts.append({"name": "mcl", "kind": "mcl", "r_p": R, "share": 1.0, "curve": False})
        elif x.isdigit() and 0 <= int(x) <= R:
            pts.append({"name": f"p{int(x)}", "kind": "wta", "r_p": int(x), "share": int(x) / R, "curve": True})
        elif x.endswith("nf") and x[:-2].isdigit() and 1 <= int(x[:-2]) <= R:
            r = int(x[:-2])
            pts.append({"name": f"p{r}nf", "kind": "wta_nofilm", "r_p": r, "share": r / R, "curve": False})
        else:
            raise SystemExit(f"unknown point {x!r} for {bench} (r_p 0..{R}, sft, {'loramcl' if bench == 'math' else 'mcl'})")
    if len({p["name"] for p in pts}) != len(pts):
        raise SystemExit("repeated points")
    for p in pts:
        R_, rp = R, p["r_p"]
        p["label"] = {"sft": "SFT (без режимов)", "existing_mcl": "LoRA-MCL (phase1_seeds, 2 ч по времени)",
                      "existing_wta1": "wta1 (фаза 1, 2 ч по времени)",
                      "mcl": "LoRA-MCL (relaxed WTA)" if bench == "mt" else "LoRA-MCL (relaxed WTA, токены wta1)",
                      "wta_nofilm": f"{rp / R_:.0%} приватной без FiLM (r_s {R_ - rp}, r_p {rp})"}.get(
            p["kind"], f"{rp / R_:.0%} приватной (r_s {R_ - rp}, r_p {rp})")
    return pts


def math_adapters(a, p, s):
    """(adapter, result.json, needs training) of a math point."""
    if p["kind"] == "existing_mcl":
        d = f"{a.seeds_run}/train/loramcl_s{s}"
        return f"{d}/adapter.pt", f"{d}/result.json", False
    if p["kind"] == "sft":
        d = f"{a.seeds_run}/train/sft1_eq_s{s}"
        if os.path.exists(f"{d}/adapter.pt") and not a.smoke:
            return f"{d}/adapter.pt", f"{d}/result.json", False
    if p["kind"] == "existing_wta1":       # phase 1 (LR and eps scheduled by time): a reference; p0 is trained anew
        return f"{wta1_dir(a, s)}/adapter.pt", f"{wta1_dir(a, s)}/result.json", False
    d = f"{a.run_abs}/train/{p['name']}_s{s}"
    return f"{d}/adapter.pt", f"{d}/result.json", True


def train_cmd(a, p, s, out, data, model):
    R, K = RANK[a.bench], MODES[a.bench]
    if p["kind"] == "sft":
        v = ["--variant", "sft", "--K", "1", "--r_s", str(R), "--r_p", "0", "--film", "0"]
    elif p["kind"] == "mcl":
        v = ["--variant", "mcl", "--K", str(K), "--r_s", "0", "--r_p", str(R), "--film", "0"]
    elif p["kind"] == "wta_nofilm":
        v = ["--variant", "wta", "--K", str(K), "--r_s", str(R - p["r_p"]), "--r_p", str(p["r_p"]), "--film", "0"]
    else:
        v = ["--variant", "wta", "--K", str(K), "--r_s", str(R - p["r_p"]), "--r_p", str(p["r_p"]), "--film", "1"]
    cmd = [PY, "-m", "phase1.ptrain", "--task", a.bench, "--base", model, "--data", data, "--out", out,
           "--seed", str(s), *v]
    if a.bench == "math":
        if a.smoke:
            cmd += ["--max_steps", "2", "--P", "4"]
        else:
            cmd += ["--target_tokens_from", f"{wta1_dir(a, s)}/result.json"]
    else:
        cmd += ["--ckpt_minutes", "0"] + (["--max_steps", "2", "--P", "8"] if a.smoke else [])
    return cmd


def est_train(a, p):
    if a.bench == "math":
        return {"sft": 1.1, "mcl": 3.0}.get(p["kind"], 2.2)
    return {"sft": 1.0, "mcl": 3.5}.get(p["kind"], 2.5) if not a.smoke else 0.1


def jobs(a, run, model, pts):
    J, P = [], []
    if a.bench == "math":
        data = f"{run}/smoke_data" if a.smoke else f"{a.phase1}/data"
        J.append(Job("refs", [PY, "-m", "phase1.peval", "refs", "--phase0", a.phase0, "--phase1", a.phase1, "--modc",
                              f"{a.seeds_run}/modc/train.jsonl", "--n_train", "8" if a.smoke else "600",
                              "--out", f"{run}/refs"], f"{run}/refs", weight=0, est=0.05))
        for p in pts:
            ent = dict(p, seeds={})
            for s in a.seeds:
                ad, res, new = math_adapters(a, p, s)
                deps = []
                if new:
                    J.append(Job(f"train_{p['name']}_s{s}", train_cmd(a, p, s, os.path.dirname(ad), data, model),
                                 os.path.dirname(ad), weight=2, est=est_train(a, p)))
                    deps = [f"train_{p['name']}_s{s}"]
                g, n = f"{run}/gen/{p['name']}_s{s}", f"{run}/nll/{p['name']}_s{s}"
                J.append(Job(f"gen_{p['name']}_s{s}", [PY, "-m", "phase1.peval", "gen", "--base", model, "--adapter", ad,
                                                       "--set_file", f"{run}/data/hv.jsonl", "--out", g, "--k", str(a.k),
                                                       "--seed", str(eval_seed(s)), "--max_tokens", str(a.max_tokens),
                                                       "--disk_root", a.disk_root, "--disk_gb", str(a.disk_gb)],
                             g, deps, weight=2, est=0.2 if p["kind"] == "sft" else 0.6))
                J.append(Job(f"nll_{p['name']}_s{s}", [PY, "-m", "phase1.peval", "nll", "--base", model, "--adapter", ad,
                                                       "--refs", f"{run}/refs/refs.jsonl.gz", "--out", n],
                             n, deps + ["refs"], weight=2, est=0.15))
                ent["seeds"][str(s)] = {"adapter": ad, "result": res, "new": new, "gen": g, "nll": n}
            P.append(ent)
        rep_deps = [j.name for j in J if j.name.startswith(("gen_", "nll_"))]
    else:
        data = f"{run}/data"
        mx = ["--max_train", "64", "--max_test", "6"] if a.smoke else []
        J.append(Job("mt_data", [PY, "-m", "phase1.pmt", "data", "--out", data, *mx], data, weight=0, est=0.05))
        base = a.mt_base or f"{run}/base"
        if not a.mt_base:
            J.append(Job("mt_base", [PY, "-m", "phase1.pmt", "base", "--out", base]
                         + (["--tmp", a.mt_tmp] if a.mt_tmp else []), base, weight=0, est=0.5))
        bdep = [] if a.mt_base else ["mt_base"]
        for p in pts:
            ent = dict(p, seeds={})
            for s in a.seeds:
                out, ev = f"{run}/train/{p['name']}_s{s}", f"{run}/eval/{p['name']}_s{s}"
                J.append(Job(f"train_{p['name']}_s{s}", train_cmd(a, p, s, out, data, base), out,
                             ["mt_data"] + bdep, weight=2, est=est_train(a, p)))
                J.append(Job(f"eval_{p['name']}_s{s}", [PY, "-m", "phase1.pmt", "eval", "--base", base, "--adapter",
                                                        f"{out}/adapter.pt", "--data", data, "--out", ev, "--beams",
                                                        a.beams, "--K", str(MODES["mt"])],
                             ev, [f"train_{p['name']}_s{s}"], weight=2, est=0.1 if a.smoke else 0.6))
                ent["seeds"][str(s)] = {"adapter": f"{out}/adapter.pt", "result": f"{out}/result.json", "new": True,
                                        "eval": ev}
            P.append(ent)
        rep_deps = [j.name for j in J if j.name.startswith("eval_")]
    J.append(Job("report", [PY, "-m", "phase1.preport", "--run", run, "--mark", f"{run}/report_job"]
                 + (["--boot", "20"] if a.smoke else []), f"{run}/report_job", rep_deps, weight=0, est=0.05, soft=True))
    return J, P


def smoke_math_data(a, run):
    d = os.path.join(run, "smoke_data")
    os.makedirs(d, exist_ok=True)
    for name, n in (("train", 24), ("verify", 24)):
        with open(os.path.join(a.phase1, "data", f"{name}.jsonl")) as f, open(os.path.join(d, f"{name}.jsonl"), "w") as g:
            g.writelines(l for i, l in enumerate(f) if i < n)


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["math", "mt"])
    ap.add_argument("--run", default="")
    ap.add_argument("--phase0", default="runs/phase0")
    ap.add_argument("--phase1", default="runs/phase1")
    ap.add_argument("--seeds_run", default="runs/phase1_seeds")
    ap.add_argument("--self_run", default="runs/phase1_self", help="math: hv.jsonl of its data")
    ap.add_argument("--model", default="", help="math: the base (default: model_path of phase 1)")
    ap.add_argument("--mt_base", default="", help="mt: a converted base directory (default: ALMA-7B-Pretrain -> <run>/base)")
    ap.add_argument("--mt_tmp", default="", help="mt: where a 9.9 GB .bin shard waits for conversion (default /dev/shm)")
    ap.add_argument("--points", default="")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--k", type=int, default=16, help="math: attempts per hv problem")
    ap.add_argument("--max_tokens", type=int, default=1200)
    ap.add_argument("--beams", default="1,3", help="mt: beam widths")
    ap.add_argument("--hours", type=float, default=6.0, help="planned time; only a warning if the plan exceeds it")
    ap.add_argument("--max_hours", type=float, default=10.0, help="jobs that would end later are skipped")
    ap.add_argument("--disk_gb", type=float, default=24.0,
                    help="disk quota of --disk_root: exports wait until they fit (the OS may report more free space)")
    ap.add_argument("--disk_root", default="", help="the tree the quota is measured on (default: the parent of --run)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--plan_only", action="store_true")
    a = ap.parse_args(argv)
    a.run = a.run or ("runs/phase1_priv" if a.bench == "math" else "runs/phase1_priv_mt")
    a.seeds = sorted({int(s) for s in a.seeds.split(",")})
    a.phase0, a.phase1, a.seeds_run, a.self_run = (os.path.abspath(p) for p in (a.phase0, a.phase1, a.seeds_run, a.self_run))
    a.mt_base = os.path.abspath(a.mt_base) if a.mt_base else ""
    a.points = a.points or DEFAULTS[a.bench]
    if a.smoke:
        a.seeds = a.seeds[:1]
        a.max_tokens, a.k = min(a.max_tokens, 256), 4
        a.points = "sft,2" if a.bench == "math" else "sft,mcl,1"
    a.run_abs = os.path.abspath(a.run)
    a.disk_root = os.path.abspath(a.disk_root or os.path.dirname(a.run_abs))
    return a


def check_inputs(a, pts):
    need = []
    if a.bench == "math":
        need += [f"{a.phase1}/data/{f}.jsonl" for f in ("train", "verify", "val")]
        need += [f"{a.self_run}/data/hv.jsonl", f"{a.phase0}/pool"]
        for p in pts:
            for s in a.seeds:
                ad, res, new = math_adapters(a, p, s)
                if not new:
                    need.append(ad)
                elif not a.smoke:
                    need.append(f"{wta1_dir(a, s)}/result.json")          # the token budget
    return [p for p in need if not os.path.exists(p)]


def main(argv=None):
    a = parse(argv)
    run = a.run_abs
    pts = parse_points(a.points, a.bench, RANK[a.bench])
    missing = check_inputs(a, pts)
    if missing:
        raise SystemExit("not found:\n  " + "\n  ".join(sorted(set(missing))))
    if a.bench == "mt":
        try:
            import sacrebleu  # noqa: F401
            import sacremoses  # noqa: F401
        except ImportError:
            raise SystemExit("mt needs sacrebleu and sacremoses: pip install sacrebleu sacremoses")
    os.makedirs(run, exist_ok=True)
    model = ""
    if a.bench == "math":
        model = a.model or json.load(open(os.path.join(a.phase1, "state.json")))["model_path"]
        if not os.path.isdir(model):
            raise SystemExit(f"model directory not found: {model} (pass --model)")
        model = os.path.abspath(model)
        os.makedirs(f"{run}/data", exist_ok=True)
        hv = f"{a.self_run}/data/hv.jsonl"
        if a.smoke:
            with open(hv) as f, open(f"{run}/data/hv.jsonl", "w") as g:
                g.writelines(l for i, l in enumerate(f) if i < 6)
            smoke_math_data(a, run)
        elif not os.path.exists(f"{run}/data/hv.jsonl"):
            shutil.copy(hv, f"{run}/data/hv.jsonl")
    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    sp = os.path.join(run, "state.json")
    state = json.load(open(sp)) if os.path.exists(sp) else {}
    fixed = {"bench": a.bench, "model_path": model, "mt_base": a.mt_base, "max_tokens": a.max_tokens, "k": a.k,
             "beams": a.beams, "smoke": a.smoke, "phase1": a.phase1, "seeds_run": a.seeds_run}
    for k, v in fixed.items():
        if state.setdefault(k, v) != v:
            raise SystemExit(f"{run} was started with {k} = {state[k]}, now {v}; use another --run")
    if a.bench == "math":           # the evaluated set and the reused adapters must not change under a finished step
        fp = state.setdefault("inputs", {})
        for f in [f"{run}/data/hv.jsonl"] + [math_adapters(a, p, s)[0] for p in pts for s in a.seeds
                                             if not math_adapters(a, p, s)[2]]:
            v = [os.path.getsize(f), int(os.path.getmtime(f))]
            if fp.setdefault(f, v) != v:
                raise SystemExit(f"{f} changed since this run started (size, mtime {fp[f]} -> {v}); use another --run")
    J, P = jobs(a, run, model, pts)
    old = json.load(open(os.path.join(run, "points.json")))["points"] if os.path.exists(os.path.join(run, "points.json")) else []
    names = {p["name"] for p in P}
    for p in old:                 # points of earlier invocations stay in the report
        if p["name"] not in names:
            P.append(p)
        else:
            cur = next(q for q in P if q["name"] == p["name"])
            for s, d in p["seeds"].items():
                cur["seeds"].setdefault(s, d)
    order = {"sft": -2.0, "wta1": -1.0, "mcl": 2.0, "loramcl": 2.1}
    P.sort(key=lambda p: order.get(p["name"], p["share"] if p["share"] is not None else -1))
    json.dump({"bench": a.bench, "points": P, "set_file": f"{run}/data/hv.jsonl", "refs": f"{run}/refs/refs.jsonl.gz",
               "data": f"{run}/data", "k": a.k, "R": RANK[a.bench], "K": MODES[a.bench]},
              open(os.path.join(run, "points.json"), "w"), indent=1)
    for j in J:
        if is_done(j.out):
            j.est = 0.0
    mk = simulate(J, gpus)
    log(f"{a.bench}: {len(J)} jobs, simulated {mk:.1f} h left on {len(gpus)} GPUs (budget {a.hours} h, hard limit "
        f"{a.max_hours} h); seeds {a.seeds}; points {[p['name'] for p in pts]}; free disk "
        f"{shutil.disk_usage(run).free / 1e9:.0f} GB (as the OS reports it)")
    if a.plan_only:
        for j in J:
            log(f"  {j.name}: est {j.est:.2f} h, deps {j.deps}")
        return
    if mk > a.hours:
        log(f"WARNING: the plan ({mk:.1f} h) is longer than the budget ({a.hours} h)")
    session = time.time()
    state.setdefault("started", session)
    state.update(session_started=session, planned_hours=mk)
    json.dump(state, open(sp, "w"), indent=1)
    shutil.rmtree(os.path.join(run, "report_job"), ignore_errors=True)
    runner = Runner(run, gpus, state, disk_limit_gb=a.disk_gb)

    def on_signal(sig, frame):
        log("signal received -- stopping children")
        runner.kill_all()
        sys.exit(1)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    ok = runner.run(J, f"phase1-priv-{a.bench}" + ("-smoke" if a.smoke else ""), deadline=session + a.max_hours * 3600)
    rp = os.path.join(run, "report_priv.md")
    log(f"finished; report: {rp}" if os.path.exists(rp) else "finished without report -- see logs/")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
