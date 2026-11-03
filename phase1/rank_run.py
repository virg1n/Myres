"""Equal trainable parameters, 2 seeds, one command: wta1 with LoRA r = 128 against LoRA-MCL (4 x r = 32), and
LoRA-MCL with r = 10 against wta1 (r = 32). The r = 32 runs of the same seeds are not repeated: they come from
phase1_seeds (seed 0 of wta1 and sftmulti from phase 1).

  wta1_r128    LoRA 128 x 1.09M + FiLM 7.6M = 147.0M    vs  loramcl r = 32: 4 x 34.9M = 139.5M
  loramcl_r10  4 x 10 x 1.09M = 43.6M                    vs  wta1 r = 32: 34.9M + 7.6M = 42.5M
  alpha        rank-stabilized scale (rsLoRA, Kalajdzievski 2023, arXiv 2312.03732): alpha / r goes as 1 / sqrt(r)
               and equals the phase-1 scale 2 at r = 32, so alpha = 2 sqrt(32 r): r = 128 -> 128 (scale 1),
               r = 10 -> 35.8 (scale 3.58). The phase-1 rule alpha = 2r keeps the scale 2 at every rank, and then
               at the same LR a larger rank also means a larger update; --alpha_rule 2r restores it.
  budget       2 h of training as before (time, not tokens: LoRA-MCL still sees ~2x fewer tokens), evaluation on
               test with the sampling seeds of phase1_seeds; on 4 GPUs ~2.1 h + ~1 h.
  flow         train x4 -> eval on test x4 -> report (phase1/rank_report.py)

Disk: last.pt with Adam moments during training (~4.6 GB in all, deleted at the end), then 4 evaluations x 3.5 GB
and the adapters ~1.5 GB: the peak is ~16 GB."""
import argparse
import json
import math
import os
import shutil
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase1.seeds_report import eval_dir  # noqa: E402
from phase1.seeds_run import PY, TRAIN_H, eval_seed, log, smoke_data  # noqa: E402
from pilot.common import is_done  # noqa: E402
from pilot.run_all import Job, Runner, simulate  # noqa: E402

VARIANTS = {"wta1_r128": ("wta1", 128), "loramcl_r10": ("loramcl", 10)}
REFS = ("wta1", "loramcl")             # the r = 32 runs every seed is compared with
EVAL_H = 1.0
R0, SCALE0 = 32, 2.0                   # phase 1: r = 32, alpha = 64


def alpha_of(r, rule):
    """LoRA alpha at rank r: "rs" keeps alpha / sqrt(r) of r = 32, "2r" keeps alpha / r."""
    return SCALE0 * math.sqrt(R0 * r) if rule == "rs" else SCALE0 * r


def jobs(a, run, data, model):
    J = []
    extra = ["--max_steps", "2", "--P", "4"] if a.smoke else []
    for s in a.seeds:
        for name, (variant, r) in VARIANTS.items():
            out = f"{run}/train/{name}_s{s}"
            J.append(Job(f"train_{name}_s{s}", [
                PY, "-m", "phase1.train", "--variant", variant, "--base", model, "--data", data, "--out", out,
                "--max_hours", f"{TRAIN_H:.2f}", "--seed", str(s), "--save_opt", "0",
                "--lora_r", str(r), "--lora_alpha", f"{alpha_of(r, a.alpha_rule):.4f}"] + extra, out, weight=2,
                est=TRAIN_H + 0.1))
    for s in a.seeds:
        for name in VARIANTS:
            J.append(Job(f"eval_{name}_s{s}", [
                PY, "-m", "phase1.evaluate", "--base", model, "--adapter", f"{run}/train/{name}_s{s}/adapter.pt",
                "--data", data, "--out", f"{run}/eval/{name}_s{s}", "--sets", "test", "--seed", str(eval_seed(s)),
                "--max_tokens", str(a.max_tokens), "--modes", "2" if a.smoke else "4"],
                f"{run}/eval/{name}_s{s}", [f"train_{name}_s{s}"], weight=2, est=EVAL_H))
    J.append(Job("report", [PY, "-m", "phase1.rank_report", "--run", run, "--data", data, "--seeds_run", a.seeds_run,
                            "--phase1", "" if a.smoke else a.phase1, "--phase0", "" if a.smoke else a.phase0,
                            "--seeds", ",".join(map(str, a.seeds)), "--mark", f"{run}/report_job"]
                 + (["--boot", "20"] if a.smoke else []),
                 f"{run}/report_job", [j.name for j in J if j.name.startswith("eval_")], weight=0, est=0.1, soft=True))
    return J


def check_inputs(a):
    """Inputs that do not exist: the data of phase 1 and the r = 32 evaluations of the planned seeds."""
    need = [f"{a.phase1}/data/{f}.jsonl" for f in ("train", "verify", "test")]
    need += [os.path.join(eval_dir(a.seeds_run, a.phase1, v, s), "eval.json") for s in a.seeds for v in REFS]
    return [p for p in need if not os.path.exists(p)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/phase1_rank")
    ap.add_argument("--seeds_run", default="runs/phase1_seeds", help="the run with the r = 32 variants of every seed")
    ap.add_argument("--phase1", default="runs/phase1")
    ap.add_argument("--phase0", default="runs/phase0")
    ap.add_argument("--model", default="")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--hours", type=float, default=3.5, help="planned time; only a warning if the plan exceeds it")
    ap.add_argument("--max_hours", type=float, default=6.0,
                    help="from the start of this invocation: jobs that would end later are skipped")
    ap.add_argument("--max_tokens", type=int, default=1200)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--alpha_rule", default="rs", choices=["rs", "2r"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--plan_only", action="store_true", help="print the jobs and the simulated time, run nothing")
    a = ap.parse_args()
    a.seeds = [1] if a.smoke else sorted({int(s) for s in a.seeds.split(",")})
    a.phase0, a.phase1, a.seeds_run = (os.path.abspath(p) for p in (a.phase0, a.phase1, a.seeds_run))
    if a.smoke:
        a.max_tokens = min(a.max_tokens, 384)
    run = os.path.abspath(a.run)
    missing = [f"{a.phase1}/data/train.jsonl"] if a.smoke and not os.path.exists(f"{a.phase1}/data/train.jsonl") \
        else [] if a.smoke else check_inputs(a)
    if missing:
        raise SystemExit("not found:\n  " + "\n  ".join(missing))
    os.makedirs(run, exist_ok=True)
    model = a.model or json.load(open(os.path.join(a.phase1, "state.json")))["model_path"]
    if not os.path.isdir(model):
        raise SystemExit(f"model directory not found: {model} (pass --model)")
    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    sp = os.path.join(run, "state.json")
    state = json.load(open(sp)) if os.path.exists(sp) else {}
    # finished steps are reused on resume: settings that change their results must stay the same
    fixed = {"alpha_rule": a.alpha_rule, "model_path": os.path.abspath(model), "max_tokens": a.max_tokens,
             "smoke": a.smoke, "seeds_run": a.seeds_run}
    for k, v in fixed.items():
        if state.setdefault(k, v) != v:
            raise SystemExit(f"{run} was started with {k} = {state[k]}, now {v}; use another --run")
    data = smoke_data(a, run) if a.smoke else os.path.join(a.phase1, "data")
    J = jobs(a, run, data, model)
    for j in J:
        if is_done(j.out):
            j.est = 0.0
    mk = simulate(J, gpus)
    alphas = {name: round(alpha_of(r, a.alpha_rule), 2) for name, (_, r) in VARIANTS.items()}
    log(f"{len(J)} jobs, simulated {mk:.1f} h left on {len(gpus)} GPUs (budget {a.hours} h, hard limit {a.max_hours} h); "
        f"seeds {a.seeds}; alpha ({a.alpha_rule}) {alphas}; free disk {shutil.disk_usage(run).free / 1e9:.0f} GB "
        f"(as the OS reports it)")
    if a.plan_only:
        for j in J:
            log(f"  {j.name}: est {j.est:.2f} h, deps {j.deps}")
        return
    if mk > a.hours:
        log(f"WARNING: the plan ({mk:.1f} h) is longer than the budget ({a.hours} h)")
    session = time.time()
    state.setdefault("started", session)
    state.update(session_started=session, planned_hours=mk, alphas=alphas)
    json.dump(state, open(sp, "w"), indent=1)
    shutil.rmtree(os.path.join(run, "report_job"), ignore_errors=True)      # the report covers what exists now
    runner = Runner(run, gpus, state, disk_limit_gb=1e9)

    def on_signal(sig, frame):
        log("signal received -- stopping children")
        runner.kill_all()
        sys.exit(1)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    ok = runner.run(J, "phase1-rank-smoke" if a.smoke else "phase1-rank", deadline=session + a.max_hours * 3600)
    rp = os.path.join(run, "report_rank.md")
    log(f"finished; report: {rp}" if os.path.exists(rp) else "finished without report -- see logs/")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
