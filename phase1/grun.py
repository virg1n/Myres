"""GRPO instead of RFT for stage 2, one command: does RL with modes keep the coverage that plain RL narrows?

  chains  per seed, from the phase-1 adapters of that seed (srun.start_adapter), phase1/grpo.py on the pool of
          phase1_self without the problems that all 8 M_0 pool attempts of sftmulti and wta1 solved:
            grpo          start sftmulti, G = 16 rollouts of one policy per problem, standard GRPO (Dr. GRPO advantage)
            grpo_wta1     start wta1, 4 rollouts per FiLM code; the same group of 16 and the same advantage, a rollout
                          trains its own code (and the shared LoRA)
            grpo_loramcl  start LoRA-MCL, 4 rollouts per adapter, as grpo_wta1 (optional: a second wave of training)
            grpo_wta1c    grpo_wta1 with --adv cover, the explicit 1/|W| (coverage) analog (ablation, optional)
          Equal for every chain of a seed: the problems and their order, G, P, the steps, LR, loss, max_tokens and the
          sampling of the rollouts; only the modes differ.
  calib   the first --calib_steps steps of every start kind (first seed, discarded) -> minutes per step; the steps S
          of every chain and seed are those the slowest kind does in --train_hours, pinned in steps.json
  evals   phase1.selftrain gen (vLLM, the sampling of phase0.gen) at the checkpoints S j / --evals: hv 16 attempts, the
          final one also test 4 -- the protocol of M_0 / M_R in phase1_self, whose M_0 generations are reused
          (runs/phase1_self/s*/start_*/gen)
  report  phase1/greport.py -> report_grpo.md
--smoke: the first seed, tiny data taken from phase1_self, its own M_0 generations, 2 steps of 2 problems x 4 rollouts.
Disk: last.pt with Adam (~0.5 GB per chain) and checkpoints (~0.15 GB each) during training, one export (3.5 GB) per
running evaluation, intermediate adapters deleted after their evaluation: the peak is ~18 GB."""
import argparse
import glob
import json
import os
import shutil
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase1.grpo import CALIB_MARGIN  # noqa: E402
from phase1.seeds_run import PY, log  # noqa: E402
from phase1.srun import EXPORTS, UNITS, start_adapter  # noqa: E402
from pilot.common import is_done  # noqa: E402
from pilot.run_all import Job, Runner, simulate  # noqa: E402

CHAINS = ("grpo", "grpo_wta1", "grpo_loramcl", "grpo_wta1c")
START = {"grpo": "sftmulti", "grpo_wta1": "wta1", "grpo_loramcl": "loramcl", "grpo_wta1c": "wta1"}
ADV = {"grpo": "mean", "grpo_wta1": "mean", "grpo_loramcl": "mean", "grpo_wta1c": "cover"}
ID = {"grpo": 1, "grpo_wta1": 2, "grpo_loramcl": 3, "grpo_wta1c": 4, "sftmulti": 5, "wta1": 6, "loramcl": 7}
SLOW = {"sftmulti": 1.0, "wta1": 1.05, "loramcl": 2.0}     # estimate: time per step against plain GRPO (loramcl: one
FILTER = ("sftmulti", "wta1")                              # generate call per adapter); M_0 of the easy-problem filter
M0_SETS = {"pool": 4, "hv": 16, "test": 4}


def eval_seed(s, j, who):
    return 50_000_000 + 1_000_000 * s + 10_000 * j + 100 * ID[who]       # mode m adds 1000 * (m + 1)


def eval_sets(j, n):
    return {"hv": 16, "test": 4} if j == n else {"hv": 16}


def m0_dir(a, run, s, st):
    return f"{run}/m0/s{s}/start_{st}/gen" if a.smoke else f"{a.self_run}/s{s}/start_{st}/gen"


def data_dir(a, run):
    return f"{run}/data" if a.smoke else f"{a.self_run}/data"


def starts(a):
    return sorted({START[c] for c in a.chains})


def sizes(a, run):
    m = os.path.join(data_dir(a, run), "meta.json")
    return json.load(open(m))["sizes"] if os.path.exists(m) else {"pool": 2400, "hv": 800, "test": 1500}


def est_gen(a, st, spec, N):
    sols = sum(n * N[k] for k, n in spec.items())
    return (sols / a.gen_per_min + UNITS[st] * a.start_min + EXPORTS[st] * a.export_min) / 60


def train_hours(a, st):
    """Planned hours of one training: fixed steps times the estimated minutes per step, or the calibrated steps, which
    fill --train_hours for the slowest start kind of the chains (a faster kind finishes earlier)."""
    if a.steps:
        return a.steps * a.step_min * SLOW[st] / 60 + 0.05
    slow = max(SLOW[x] for x in starts(a))
    return min(a.train_hours, a.steps_max * a.step_min * slow / 60) * SLOW[st] / slow + 0.05


def planned_steps(a, run):
    """(S, source) for the plan: --steps, the pinned steps.json, or the estimate --step_min."""
    if a.steps:
        return a.steps, "--steps"
    f = os.path.join(run, "steps.json")
    if os.path.exists(f):
        return json.load(open(f))["steps"], "steps.json"
    mps = a.step_min * max(SLOW[st] for st in starts(a))
    S = min(a.steps_max, int(a.train_hours * 60 / (CALIB_MARGIN * mps)))
    return max(a.evals, S // a.evals * a.evals), f"estimate {mps:.2f} min/step"


def jobs(a, run, model, report=None):
    """report: (seeds, chains) of the report, all of the run so far (state.json); default those of this invocation."""
    J = []
    data = data_dir(a, run)
    N = sizes(a, run)
    calib = not a.steps and not os.path.exists(os.path.join(run, "steps.json"))
    gargs = ["--base", model, "--data", data, "--G", str(a.G), "--P", str(a.P), "--evals", str(a.evals), "--lr",
             str(a.lr), "--film_lr_ratio", str(a.film_lr_ratio), "--warmup", str(a.warmup), "--temp", str(a.temp),
             "--max_tokens", str(a.max_tokens), "--gen_batch", str(a.gen_batch), "--gen_tokens", str(a.gen_tokens),
             "--cache", a.cache, "--drop_easy", str(a.drop_easy)]
    if a.smoke:
        for si, s in enumerate(a.seeds):
            for st in sorted(set(starts(a)) | set(FILTER)):
                out = m0_dir(a, run, s, st)
                J.append(Job(f"m0_s{s}_{st}", [PY, "-m", "phase1.selftrain", "gen", "--base", model, "--adapter",
                                               start_adapter(a, s, st), "--data", data, "--out", out, "--sets",
                                               ",".join(f"{k}:{n}" for k, n in M0_SETS.items()), "--seed",
                                               str(eval_seed(s, 0, st)), "--max_tokens", str(a.max_tokens)],
                             out, weight=2, est=est_gen(a, st, M0_SETS, N), prio=60))
    m0deps = lambda s: [j.name for j in J if j.name.startswith(f"m0_s{s}_")]
    m0 = lambda s: ",".join(m0_dir(a, run, s, st) for st in FILTER)
    cfiles = [f"{run}/calib/{st}/calib.json" for st in starts(a)]
    if calib:
        s = a.seeds[0]
        for st in starts(a):
            c = next(c for c in a.chains if START[c] == st)
            J.append(Job(f"calib_{st}", [PY, "-m", "phase1.grpo", *gargs, "--init", start_adapter(a, s, st), "--m0", m0(s),
                                         "--out", f"{run}/calib/{st}", "--seed", str(s), "--adv", ADV[c],
                                         "--calib", str(a.calib_steps)],
                         f"{run}/calib/{st}", m0deps(s), weight=2,
                         est=(3 + a.calib_steps * a.step_min * SLOW[st]) / 60, prio=50))
    steps = ["--steps", str(a.steps)] if a.steps else \
        ["--steps_file", f"{run}/steps.json", "--calib_from", ",".join(cfiles), "--train_hours", str(a.train_hours),
         "--steps_max", str(a.steps_max)]
    for si, s in enumerate(a.seeds):
        for c in a.chains:
            st, d = START[c], f"{run}/s{s}/{c}"
            h = train_hours(a, st)
            safety = 1.5 * (h if a.steps else a.train_hours * SLOW[st]) + 0.5      # far above a normal training
            J.append(Job(f"train_s{s}_{c}", [PY, "-m", "phase1.grpo", *gargs, *steps, "--init", start_adapter(a, s, st),
                                             "--m0", m0(s), "--out", f"{d}/train", "--seed", str(s), "--adv", ADV[c],
                                             "--max_hours", f"{safety:.2f}"],
                         f"{d}/train", [j.name for j in J if j.name.startswith("calib_")] + m0deps(s), weight=2,
                         est=h, prio=40 - si))
            for j in range(1, a.evals + 1):
                spec = eval_sets(j, a.evals)
                J.append(Job(f"gen_s{s}_{c}_c{j}", [
                    PY, "-m", "phase1.selftrain", "gen", "--base", model, "--adapter", f"{d}/train/ckpt{j}/adapter.pt",
                    "--data", data, "--out", f"{d}/c{j}/gen", "--sets", ",".join(f"{k}:{n}" for k, n in spec.items()),
                    "--seed", str(eval_seed(s, j, c)), "--max_tokens", str(a.max_tokens),
                    "--drop_adapter", str(int(j < a.evals))],
                    f"{d}/c{j}/gen", [f"train_s{s}_{c}"], weight=2, est=est_gen(a, st, spec, N),
                    prio=(30 if j == a.evals else 20) - si))
    rs, rc = report or (a.seeds, a.chains)
    rep = [PY, "-m", "phase1.greport", "--run", run, "--data", data, "--m0", f"{run}/m0" if a.smoke else a.self_run,
           "--self_run", "" if a.smoke else a.self_run, "--seeds", ",".join(map(str, rs)),
           "--chains", ",".join(rc), "--evals", str(a.evals), "--mark", f"{run}/report_job"]
    J.append(Job("report", rep + (["--boot", "20"] if a.smoke else []), f"{run}/report_job",
                 [j.name for j in J if j.name.startswith("gen_s")], weight=0, est=0.1, soft=True))
    return J


def same_path(x, y):
    return bool(x) and os.path.exists(x) and os.path.exists(y) and os.path.samefile(x, y)


def check_inputs(a, model):
    """Missing inputs, and M_0 generations that are not of the start adapter (path in gen.json; size and mtime in
    keval.json), of another base, data or max_tokens, or not of the needed sets."""
    need = [] if a.smoke else [f"{a.self_run}/data/{f}" for f in ("pool.jsonl", "hv.jsonl", "test.jsonl", "meta.json")]
    if a.smoke:
        need += [f"{a.self_run}/data/{f}.jsonl" for f in ("pool", "hv", "test")]
    need += sorted({start_adapter(a, s, st) for s in a.seeds for st in set(starts(a)) | set(FILTER)})
    bad = [p for p in need if not os.path.exists(p)]
    if a.smoke:
        return bad
    for s in a.seeds:
        for st in sorted(set(starts(a)) | set(FILTER)):
            g = m0_dir(a, None, s, st)
            if not os.path.exists(os.path.join(g, "DONE")):
                bad.append(f"{g} (M_0 of phase1_self, not finished)")
                continue
            man = json.load(open(os.path.join(g, "gen.json")))
            if any(man["sets"].get(k, 0) < n for k, n in M0_SETS.items()):
                bad.append(f"{g}: sets {man['sets']}, need {M0_SETS}")
            ad = start_adapter(a, s, st)
            if os.path.exists(man.get("adapter", "")) and os.path.exists(ad) and not os.path.samefile(man["adapter"], ad):
                bad.append(f"{g}: generated by {man['adapter']}, not by the start adapter {ad}")
            kf = os.path.join(g, "keval.json")
            if not os.path.exists(kf):
                bad.append(f"{g}: no keval.json (the settings of the M_0 generation)")
                continue
            k = json.load(open(kf))
            if os.path.exists(ad) and [k.get("adapter_bytes"), k.get("adapter_mtime")] != \
                    [os.path.getsize(ad), int(os.path.getmtime(ad))]:
                bad.append(f"{g}: generated by another version of {ad} (size, mtime {k.get('adapter_bytes')}, "
                           f"{k.get('adapter_mtime')})")
            if k.get("max_tokens") != a.max_tokens:
                bad.append(f"{g}: max_tokens {k.get('max_tokens')}, this run {a.max_tokens}")
            if not same_path(k.get("base"), model):
                bad.append(f"{g}: base {k.get('base')}, this run {model}")
            if not same_path(k.get("data"), f"{a.self_run}/data"):
                bad.append(f"{g}: data {k.get('data')}, this run {a.self_run}/data")
    return bad


def smoke_data(a, run):
    """A few problems of each set of phase1_self (their M_0 comes from the smoke run's own generations)."""
    from phase1.train import load_jsonl
    d = data_dir(a, run)
    os.makedirs(d, exist_ok=True)
    n = {"pool": 6, "hv": 4, "test": 4}
    for name, k in n.items():
        rows = load_jsonl(f"{a.self_run}/data/{name}.jsonl")[:k]
        with open(f"{d}/{name}.jsonl", "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    json.dump({"sizes": n, "smoke": True}, open(f"{d}/meta.json", "w"))


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/phase1_grpo")
    ap.add_argument("--self_run", default="runs/phase1_self", help="data, M_0 generations and the RFT chains")
    ap.add_argument("--phase1", default="runs/phase1")
    ap.add_argument("--seeds_run", default="runs/phase1_seeds", help="loramcl of seed 0, all start adapters of seed > 0")
    ap.add_argument("--model", default="")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--chains", default="grpo,grpo_wta1", help=f"from {CHAINS}")
    ap.add_argument("--steps", type=int, default=0, help="optimizer steps of every chain; 0: by calibration")
    ap.add_argument("--train_hours", type=float, default=9.0, help="calibrated steps: the slowest chain fits this")
    ap.add_argument("--steps_max", type=int, default=1000)
    ap.add_argument("--calib_steps", type=int, default=4, help="steps of a calibration run (the first one not counted)")
    ap.add_argument("--evals", type=int, default=3, help="evaluated checkpoints at S j / evals")
    ap.add_argument("--G", type=int, default=16)
    ap.add_argument("--P", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--film_lr_ratio", type=float, default=5.0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--temp", type=float, default=1.0, help="rollout temperature (evaluations: phase0.gen sampling)")
    ap.add_argument("--max_tokens", type=int, default=1200)
    ap.add_argument("--gen_batch", type=int, default=128)
    ap.add_argument("--gen_tokens", type=int, default=128 * 1800, help="KV cache positions per generate call")
    ap.add_argument("--cache", choices=["dynamic", "static"], default="dynamic")
    ap.add_argument("--drop_easy", type=int, default=1)
    ap.add_argument("--hours", type=float, default=14.0, help="planned time; only a warning if the plan exceeds it")
    ap.add_argument("--max_hours", type=float, default=16.0,
                    help="from the start of this invocation: jobs that would end later are skipped")
    ap.add_argument("--step_min", type=float, default=1.6, help="estimate: minutes per step of plain GRPO (128 rollouts)")
    ap.add_argument("--gen_per_min", type=float, default=500, help="estimate: vLLM solutions per minute per GPU")
    ap.add_argument("--start_min", type=float, default=1.0, help="estimate: vLLM start per checkpoint, minutes")
    ap.add_argument("--export_min", type=float, default=2.0, help="estimate: one full export, minutes")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--plan_only", action="store_true", help="print the jobs and the simulated time, run nothing")
    a = ap.parse_args(argv)
    a.seeds = [int(s) for s in a.seeds.split(",")]
    a.chains = [c for c in a.chains.split(",") if c]
    if any(c not in CHAINS for c in a.chains) or len(set(a.seeds)) != len(a.seeds) or not a.chains:
        raise SystemExit(f"--chains from {CHAINS}, distinct --seeds")
    a.self_run, a.phase1, a.seeds_run = map(os.path.abspath, (a.self_run, a.phase1, a.seeds_run))
    if a.smoke:                  # every kind of job once: the first seed, calibration, 2 steps, 2 checkpoints
        a.seeds = a.seeds[:1]
        a.max_tokens, a.G, a.P, a.steps, a.steps_max, a.evals = min(a.max_tokens, 256), 4, 2, 0, 2, 2
        a.calib_steps, a.train_hours = 2, 1.0
    if a.G % 4 or a.steps and a.steps < a.evals:
        raise SystemExit("--G must be a multiple of the 4 modes, --steps at least --evals")
    if a.temp != 1:
        raise SystemExit("--temp 1: the GRPO loss differentiates log pi of the policy that samples the rollouts")
    return a


def main():
    a = parse()
    run = os.path.abspath(a.run)
    model = a.model or json.load(open(os.path.join(a.phase1, "state.json")))["model_path"]
    if not os.path.isdir(model):
        raise SystemExit(f"model directory not found: {model} (pass --model)")
    missing = check_inputs(a, model)
    if missing:
        raise SystemExit("not found or not usable:\n  " + "\n  ".join(missing))
    os.makedirs(run, exist_ok=True)
    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    sp = os.path.join(run, "state.json")
    state = json.load(open(sp)) if os.path.exists(sp) else {}
    # finished steps are reused on resume: settings that change their results stay the same (seeds and chains may be
    # added; the pinned steps.json keeps S of the chains trained before)
    fixed = {"model_path": os.path.abspath(model), "max_tokens": a.max_tokens, "steps": a.steps,
             "train_hours": a.train_hours, "steps_max": a.steps_max, "calib_steps": a.calib_steps, "evals": a.evals,
             "G": a.G, "P": a.P, "lr": a.lr, "film_lr_ratio": a.film_lr_ratio, "warmup": a.warmup, "temp": a.temp,
             "gen_batch": a.gen_batch, "gen_tokens": a.gen_tokens, "cache": a.cache, "drop_easy": a.drop_easy,
             "smoke": a.smoke,
             "self_run": a.self_run, "phase1": a.phase1, "seeds_run": a.seeds_run}
    for k, v in fixed.items():
        if state.setdefault(k, v) != v:
            raise SystemExit(f"{run} was started with {k} = {state[k]}, now {v}; use another --run")
    inputs = sorted({start_adapter(a, s, st) for s in a.seeds for st in set(starts(a)) | set(FILTER)} |
                    {f"{a.self_run}/data/{f}.jsonl" for f in ("pool", "hv", "test")} |
                    (set() if a.smoke else {f"{m0_dir(a, run, s, st)}/gen.json" for s in a.seeds
                                            for st in set(starts(a)) | set(FILTER)}))
    fp = state.setdefault("inputs", {})
    for f in inputs:
        v = [os.path.getsize(f), int(os.path.getmtime(f))]
        if fp.setdefault(f, v) != v:
            raise SystemExit(f"{f} changed since this run started (size, mtime {fp[f]} -> {v}); use another --run")
    state["seeds"] = sorted(set(state.get("seeds", [])) | set(a.seeds))
    state["chains"] = [c for c in CHAINS if c in set(state.get("chains", [])) | set(a.chains)]
    if a.smoke and not os.path.exists(os.path.join(run, "data", "meta.json")):
        smoke_data(a, run)
    J = jobs(a, run, model, (state["seeds"], state["chains"]))
    for j in J:
        if is_done(j.out):
            j.est = 0.0
    mk = simulate(J, gpus)
    S, src = planned_steps(a, run)
    log(f"{len(J)} jobs, simulated {mk:.1f} h left on {len(gpus)} GPUs (budget {a.hours} h, hard limit {a.max_hours} h); "
        f"seeds {a.seeds}, chains {a.chains}; {S} steps ({src}) x {a.P} problems x {a.G} rollouts = {S * a.P * a.G} "
        f"rollouts per chain, {a.evals} checkpoints; free disk {shutil.disk_usage(run).free / 1e9:.0f} GB "
        f"(as the OS reports it)")
    if a.plan_only:
        for j in J:
            log(f"  {j.name}: est {j.est:.2f} h, prio {j.prio}, deps {j.deps}")
        return
    if mk > a.hours:
        log(f"WARNING: the plan ({mk:.1f} h) is longer than the budget ({a.hours} h): lower --train_hours")
    session = time.time()
    state.setdefault("started", session)
    state.update(session_started=session, planned_hours=mk)
    json.dump(state, open(sp, "w"), indent=1)
    shutil.rmtree(os.path.join(run, "report_job"), ignore_errors=True)      # the report covers what exists now
    # exports of evaluations killed with the previous invocation (a killed process skips its finally); nothing runs
    # now: run_phase1_grpo.sh holds the lock of the run directory
    for t in glob.glob(os.path.join(run, "**", "tmp_export"), recursive=True):
        shutil.rmtree(t, ignore_errors=True)
    runner = Runner(run, gpus, state, disk_limit_gb=1e9)

    def on_signal(sig, frame):
        log("signal received -- stopping children")
        runner.kill_all()
        sys.exit(1)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    ok = runner.run(J, "phase1-grpo-smoke" if a.smoke else "phase1-grpo", deadline=session + a.max_hours * 3600)
    rp = os.path.join(run, "report_grpo.md")
    log(f"finished; report: {rp}" if os.path.exists(rp) else "finished without report -- see logs/")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
