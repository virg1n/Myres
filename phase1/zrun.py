"""Continuous random z instead of 4 learned codes, seed 0, one command (~5 h on 4 GPUs).

  wtaz_d{8,16,128,1024}  wta1 with z = N(0, I_d) / sqrt(d) drawn anew: every example tries 4 fresh z, the one with
                         the lowest loss trains (phase1/train.py --variant wtaz); LoRA r = 32 + FiLM, 2 h, seed 0;
                         inside FiLM z is scaled by 4 with the first layer of wta1, so d = 16 is wta1 with fresh codes
  evaluation             test, k = 32 attempts per problem, each with its own z, greedy for 8 z (phase1/keval.py)
  references, seed 0     wta1 and sftmulti of phase 1, loramcl of phase1_seeds, evaluated at k = 32 the same way:
                         8 samples per mode / 32 samples, greedy per mode
  flow                   train x4 -> keval x4; keval of the references x3 on free GPUs -> report (phase1/zreport.py)

Disk: last.pt during training (~0.5 GB each), then one shared export (3.5 GB) per running evaluation of a FiLM
model, one exported adapter at a time for loramcl and sftmulti: the peak is ~15 GB."""
import argparse
import json
import os
import shutil
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase1.seeds_run import PY, TRAIN_H, eval_seed, log, smoke_data  # noqa: E402
from pilot.common import is_done  # noqa: E402
from pilot.run_all import Job, Runner, simulate  # noqa: E402

ZDIMS = (8, 16, 128, 1024)
N_TEST = 1500                 # problems in the phase-1 test (estimates only)
START_MIN = 1.0               # vLLM start of one checkpoint, minutes
GEN_MIN = 1.7                 # one solution for each of N_TEST problems, minutes (sampled or greedy)
EXPORT_MIN = 2.0              # full export of an adapter, minutes


def refs(a):
    """reference -> (adapter, result.json of its training): seed 0 of the earlier runs."""
    return {"wta1": f"{a.phase1}/train/wta1", "sftmulti": f"{a.phase1}/train/sftmulti",
            "loramcl": f"{a.seeds_run}/train/loramcl_s0"}


def est_h(kind, k, greedy, n_test):
    """Hours of one keval job: vLLM starts, exports and generation (k sampled + greedy solutions per problem)."""
    f = n_test / N_TEST
    if kind == "z":
        return (k * START_MIN + (k + greedy) * GEN_MIN * f) / 60
    if kind == "modes":
        return (4 * (START_MIN + EXPORT_MIN) + (k + 4) * GEN_MIN * f) / 60
    return (START_MIN + EXPORT_MIN + (k + 1) * GEN_MIN * f) / 60


def jobs(a, run, data, model):
    J = []
    extra = ["--max_steps", "2", "--P", "4"] if a.smoke else []
    kv = ["--k", str(a.k), "--greedy", str(a.greedy), "--seed", str(eval_seed(0)), "--max_tokens", str(a.max_tokens)]
    n_test = sum(1 for _ in open(os.path.join(data, "test.jsonl")))
    for zd in ZDIMS:
        J.append(Job(f"train_wtaz_d{zd}", [
            PY, "-m", "phase1.train", "--variant", "wtaz", "--zdim", str(zd), "--base", model, "--data", data,
            "--out", f"{run}/train/wtaz_d{zd}", "--max_hours", f"{TRAIN_H:.2f}", "--seed", "0", "--save_opt", "0"]
            + extra, f"{run}/train/wtaz_d{zd}", weight=2, est=TRAIN_H + 0.1))
    for zd in ZDIMS:
        J.append(Job(f"keval_wtaz_d{zd}", [
            PY, "-m", "phase1.keval", "--base", model, "--adapter", f"{run}/train/wtaz_d{zd}/adapter.pt",
            "--data", data, "--out", f"{run}/eval/wtaz_d{zd}"] + kv, f"{run}/eval/wtaz_d{zd}",
            [f"train_wtaz_d{zd}"], weight=2, est=est_h("z", a.k, a.greedy, n_test)))
    for name, d in refs(a).items():
        J.append(Job(f"keval_{name}", [
            PY, "-m", "phase1.keval", "--base", model, "--adapter", f"{d}/adapter.pt", "--data", data,
            "--out", f"{run}/eval/{name}"] + kv, f"{run}/eval/{name}", weight=2,
            est=est_h("single" if name == "sftmulti" else "modes", a.k, a.greedy, n_test)))
    J.append(Job("report", [PY, "-m", "phase1.zreport", "--run", run, "--data", data, "--base", model,
                            "--phase1", a.phase1, "--seeds_run", a.seeds_run, "--mark", f"{run}/report_job"]
                 + (["--boot", "20"] if a.smoke else []),
                 f"{run}/report_job", [j.name for j in J if j.name.startswith("keval_")], weight=0, est=0.1, soft=True))
    return J


def check_inputs(a):
    """Inputs that do not exist: the data of phase 1 and the trained references of seed 0."""
    need = [f"{a.phase1}/data/{f}.jsonl" for f in ("train", "verify", "test")]
    need += [f"{d}/{f}" for d in refs(a).values() for f in ("adapter.pt", "result.json")]
    return [p for p in need if not os.path.exists(p)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/phase1_z")
    ap.add_argument("--seeds_run", default="runs/phase1_seeds", help="the run with loramcl of seed 0")
    ap.add_argument("--phase1", default="runs/phase1")
    ap.add_argument("--model", default="")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--hours", type=float, default=5.5, help="planned time; only a warning if the plan exceeds it")
    ap.add_argument("--max_hours", type=float, default=7.0,
                    help="from the start of this invocation: jobs that would end later are skipped")
    ap.add_argument("--max_tokens", type=int, default=1200)
    ap.add_argument("--k", type=int, default=32, help="attempts per problem (a multiple of 4)")
    ap.add_argument("--greedy", type=int, default=8, help="greedy solutions with different z")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--plan_only", action="store_true", help="print the jobs and the simulated time, run nothing")
    a = ap.parse_args()
    a.phase1, a.seeds_run = os.path.abspath(a.phase1), os.path.abspath(a.seeds_run)
    if a.smoke:
        a.max_tokens, a.k, a.greedy = min(a.max_tokens, 384), 4, 2
    if a.k % 4 or a.greedy > a.k:
        raise SystemExit("--k must be a multiple of 4 and --greedy at most --k")
    run = os.path.abspath(a.run)
    missing = check_inputs(a)
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
    fixed = {"model_path": os.path.abspath(model), "max_tokens": a.max_tokens, "k": a.k, "greedy": a.greedy,
             "smoke": a.smoke, "seeds_run": a.seeds_run, "phase1": a.phase1}
    for k, v in fixed.items():
        if state.setdefault(k, v) != v:
            raise SystemExit(f"{run} was started with {k} = {state[k]}, now {v}; use another --run")
    data = smoke_data(a, run) if a.smoke else os.path.join(a.phase1, "data")
    J = jobs(a, run, data, model)
    for j in J:
        if is_done(j.out):
            j.est = 0.0
    mk = simulate(J, gpus)
    log(f"{len(J)} jobs, simulated {mk:.1f} h left on {len(gpus)} GPUs (budget {a.hours} h, hard limit {a.max_hours} h); "
        f"zdim {list(ZDIMS)}, k = {a.k}, greedy {a.greedy}; free disk {shutil.disk_usage(run).free / 1e9:.0f} GB "
        f"(as the OS reports it)")
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
    shutil.rmtree(os.path.join(run, "report_job"), ignore_errors=True)      # the report covers what exists now
    runner = Runner(run, gpus, state, disk_limit_gb=1e9)

    def on_signal(sig, frame):
        log("signal received -- stopping children")
        runner.kill_all()
        sys.exit(1)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    ok = runner.run(J, "phase1-z-smoke" if a.smoke else "phase1-z", deadline=session + a.max_hours * 3600)
    rp = os.path.join(run, "report_z.md")
    log(f"finished; report: {rp}" if os.path.exists(rp) else "finished without report -- see logs/")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
