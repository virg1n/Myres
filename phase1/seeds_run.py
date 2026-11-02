"""Seeds and near methods with one command: no new generation, data of phase 1, resumable.

  variants  wta1, wtabal, sftmulti (2 h each, as in phase 1), loramcl, modc, sftmodc (2 h each),
            sft1_eq = the references of wta1, up to the training tokens of wta1 of the same seed, with its
            own full LR schedule;
            sftmulti_eq = sftmulti limited by tokens: a snapshot of the 2-hour run at the tokens of wta1
            of seed 0 (other examples and an unfinished LR schedule, an additional control only)
  seeds     0, 1, 2; seed 0 of wta1, wtabal, sftmulti is taken from phase 1 as it is; there sftmulti_eq
            stops at the snapshot
            --seeds 0,1,2,3 adds seed 3 with all variants to the same run directory
  flow      modc_data -> train x19 -> eval on test x21 -> report (phase1/seeds_report.py)

--max_hours counts from the start of this invocation, so a resumed run gets a new deadline; the report
is rebuilt by every invocation.

Disk: at most one exported checkpoint (3.5 GB) per running evaluation, adapters are saved without Adam
moments; the peak is ~19-21 GB."""
import argparse
import json
import os
import shutil
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pilot.common import is_done  # noqa: E402
from pilot.run_all import Job, Runner, simulate  # noqa: E402

PY = sys.executable
TRAIN_H = 2.0
EQ_H = 0.9            # estimate: a run stopped at the tokens of wta1
OLD = ["wta1", "wtabal", "sftmulti"]
NEW = ["loramcl", "modc", "sftmodc", "sft1_eq"]
SINGLE = {"sftmulti", "sftmodc", "sft1_eq", "sftmulti_eq"}
EVAL_H = {"single": 0.5, "modc": 0.7, "modes": 1.0}
SMOKE_NO_EVAL = {"wtabal", "sftmodc"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def eval_seed(s):
    """Sampling seed of the evaluation; seed 0 = what phase 1 used (mode k adds 1000*(k+1))."""
    return 17 + 10000 * s


def train_plan(smoke, seeds=(0, 1, 2)):
    if smoke:
        return {0: ["sftmulti_eq"], 1: OLD + NEW}
    return {s: NEW + ["sftmulti_eq"] if s == 0 else OLD + NEW for s in seeds}      # OLD first: sft1_eq needs wta1 of its seed


def eq_adapter(run, s):
    """sftmulti_eq: seed 0 is its own run stopped at the snapshot, others are snapshots of sftmulti."""
    if s == 0:
        return f"{run}/train/sftmulti_eq_s0/adapter.pt", "train_sftmulti_eq_s0"
    return f"{run}/train/sftmulti_s{s}/snap/adapter.pt", f"train_sftmulti_s{s}"


def jobs(a, run, data, model, tokens):
    sm = a.smoke
    J = [Job("modc_data", [PY, "-m", "phase1.modc_data", "--data", data, "--pools", f"{a.phase0}/pool",
                           f"{a.phase1}/hard/out", "--out", f"{run}/modc"], f"{run}/modc", weight=0, est=0.05)]
    extra = ["--max_steps", "2", "--P", "4"] if sm else []
    plan = train_plan(sm, a.seeds)
    for s, variants in plan.items():
        for v in variants:
            variant = {"sftmulti_eq": "sftmulti", "sft1_eq": "sft1"}.get(v, v)
            cmd = [PY, "-m", "phase1.train", "--variant", variant, "--base", model,
                   "--data", data, "--out", f"{run}/train/{v}_s{s}", "--max_hours", f"{TRAIN_H:.2f}", "--seed", str(s),
                   "--save_opt", "0"] + extra
            deps, est = [], TRAIN_H + 0.1
            if v in ("modc", "sftmodc"):
                cmd += ["--train_file", f"{run}/modc/train.jsonl"]
                deps = ["modc_data"]
            if v == "sftmulti_eq":
                cmd += ["--snap_tokens", str(tokens), "--stop_after_snap"]
                est = EQ_H
            elif v == "sft1_eq":
                old = s == 0 and "wta1" not in variants
                cmd += ["--target_tokens_from", f"{a.phase1}/train/wta1/result.json" if old
                        else f"{run}/train/wta1_s{s}/result.json"]
                deps, est = ([] if old else [f"train_wta1_s{s}"]), EQ_H
            elif v == "sftmulti":
                cmd += ["--snap_tokens", str(tokens)]
            J.append(Job(f"train_{v}_s{s}", cmd, f"{run}/train/{v}_s{s}", deps, weight=2, est=est))
    mt = ["--max_tokens", str(a.max_tokens), "--modes", "2" if sm else "4"]

    def ev(v, s, adapter, dep, kind):
        J.append(Job(f"eval_{v}_s{s}", [PY, "-m", "phase1.evaluate", "--base", model, "--adapter", adapter,
                                        "--data", data, "--out", f"{run}/eval/{v}_s{s}", "--sets", "test",
                                        "--seed", str(eval_seed(s))] + mt,
                     f"{run}/eval/{v}_s{s}", [dep], weight=2, est=EVAL_H[kind]))

    for s, variants in plan.items():
        for v in variants:
            if sm and v in SMOKE_NO_EVAL:
                continue
            if v == "sftmulti_eq":
                continue
            ev(v, s, f"{run}/train/{v}_s{s}/adapter.pt", f"train_{v}_s{s}",
               "single" if v in SINGLE else "modc" if v == "modc" else "modes")
        if "sftmulti_eq" in variants or "sftmulti" in variants:
            ev("sftmulti_eq", s, *eq_adapter(run, s), "single")
    J.append(Job("report", [PY, "-m", "phase1.seeds_report", "--run", run, "--data", data,
                            "--phase1", "" if sm else a.phase1, "--phase0", "" if sm else a.phase0,
                            "--mark", f"{run}/report_job"] + (["--boot", "20"] if sm else []),
                 f"{run}/report_job", [j.name for j in J if j.name.startswith("eval_")], weight=0, est=0.1, soft=True))
    return J


def check_inputs(a):
    """Paths taken from the earlier phases that do not exist."""
    need = [f"{a.phase1}/data/{f}.jsonl" for f in ("train", "verify", "test")] + \
           [f"{a.phase1}/train/wta1/result.json", f"{a.phase0}/test/verify_strategies.json",
            f"{a.phase0}/test/verify_neutral.json", f"{a.phase0}/pool", f"{a.phase1}/hard/out"] + \
           [f"{a.phase0}/test/test_strategy{k}.jsonl.gz" for k in range(4)]
    for v in OLD:
        need.append(f"{a.phase1}/eval/{v}/eval.json")
    return [p for p in need if not os.path.exists(p)]


def smoke_data(a, run):
    d = os.path.join(run, "smoke_data")
    os.makedirs(d, exist_ok=True)
    for name, n in (("train", 24), ("verify", 24), ("test", 16)):
        with open(os.path.join(a.phase1, "data", f"{name}.jsonl")) as f, open(os.path.join(d, f"{name}.jsonl"), "w") as g:
            g.writelines(l for i, l in enumerate(f) if i < n)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/phase1_seeds")
    ap.add_argument("--phase1", default="runs/phase1")
    ap.add_argument("--phase0", default="runs/phase0")
    ap.add_argument("--model", default="")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--hours", type=float, default=14.0, help="planned time; only a warning if the plan exceeds it")
    ap.add_argument("--max_hours", type=float, default=16.0,
                    help="from the start of this invocation: jobs that would end later are skipped")
    ap.add_argument("--max_tokens", type=int, default=1200)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--plan_only", action="store_true", help="print the jobs and the simulated time, run nothing")
    ap.add_argument("--seeds", default="0,1,2", help="finished seeds are skipped: 0,1,2,3 adds seed 3 to a finished run")
    a = ap.parse_args()
    a.seeds = sorted({int(s) for s in a.seeds.split(",")})
    a.phase0, a.phase1 = os.path.abspath(a.phase0), os.path.abspath(a.phase1)
    if a.smoke:
        a.max_tokens = min(a.max_tokens, 384)
    run = os.path.abspath(a.run)
    missing = [] if a.smoke else check_inputs(a)
    if a.smoke and not os.path.exists(f"{a.phase1}/data/train.jsonl"):
        missing = [f"{a.phase1}/data/train.jsonl"]
    if missing:
        raise SystemExit("not found:\n  " + "\n  ".join(missing))
    os.makedirs(run, exist_ok=True)
    model = a.model
    if not model:
        model = json.load(open(os.path.join(a.phase1, "state.json")))["model_path"]
    if not os.path.isdir(model):
        raise SystemExit(f"model directory not found: {model} (pass --model)")
    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    sp = os.path.join(run, "state.json")
    state = json.load(open(sp)) if os.path.exists(sp) else {}
    if "eq_tokens" not in state:
        if a.smoke:
            state["eq_tokens"] = 1
        else:
            ref = json.load(open(f"{a.phase1}/train/wta1/result.json"))
            state["eq_tokens"] = ref["tokens"]
            if abs(ref.get("time_limit_h", TRAIN_H) - TRAIN_H) > 0.01:
                log(f"WARNING: wta1 of phase 1 had a limit of {ref['time_limit_h']} h, here {TRAIN_H} h")
    data = smoke_data(a, run) if a.smoke else os.path.join(a.phase1, "data")
    J = jobs(a, run, data, model, state["eq_tokens"])
    for j in J:
        if is_done(j.out):         # the plan of a resumed run counts what is left
            j.est = 0.0
    mk = simulate(J, gpus)
    free = shutil.disk_usage(run).free / 1e9
    log(f"{len(J)} jobs, simulated {mk:.1f} h left on {len(gpus)} GPUs (budget {a.hours} h, hard limit {a.max_hours} h); "
        f"eq tokens {state['eq_tokens'] / 1e6:.1f}M; free disk {free:.0f} GB")
    if a.plan_only:
        for j in J:
            log(f"  {j.name}: est {j.est:.2f} h, deps {j.deps}")
        return
    if mk > a.hours:
        log(f"WARNING: the plan ({mk:.1f} h) is longer than the budget ({a.hours} h)")
    if free < 20 and not a.smoke:
        log(f"WARNING: {free:.0f} GB free, the peak is ~19-21 GB; evaluations will wait for disk if needed")
    session = time.time()
    state.setdefault("started", session)
    state.update(session_started=session, model_path=model, planned_hours=mk)
    json.dump(state, open(sp, "w"), indent=1)
    shutil.rmtree(os.path.join(run, "report_job"), ignore_errors=True)      # the report covers what exists now
    runner = Runner(run, gpus, state, disk_limit_gb=1e9)

    def on_signal(sig, frame):
        log("signal received -- stopping children")
        runner.kill_all()
        sys.exit(1)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    ok = runner.run(J, "phase1-seeds-smoke" if a.smoke else "phase1-seeds",
                    deadline=session + a.max_hours * 3600)
    rp = os.path.join(run, "report_seeds.md")
    log(f"finished; report: {rp}" if os.path.exists(rp) else "finished without report -- see logs/")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
