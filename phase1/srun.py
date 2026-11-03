"""Multi-round self-training with one command: do the modes find and keep more hard problems than RFT and LoRA-MCL?

  pool      phase1/spool.py: problems without a correct solution in phase-1 training (0-correct problems of the old
            pools + fresh MATH / synthetic problems screened by the base model, 8 samples) -> hv (held-out hard
            problems, never trained on), pool (hard + medium), frozen (600 pool-hard problems)
  chains    per seed, from the phase-1 adapters of that seed (STaR / ReST-EM: every round restarts from the start
            adapter and trains on the bank of everything found so far, fixed producer routing):
              rft      start sftmulti, 4 samples per problem, every bank solution weight 1
              rftbal   start sftmulti, the same samples, each solved problem weight 1 (1 / occupied slots)
              wta1     start wta1 (FiLM codes), 1 sample per mode, a solution trains the mode that found it, 1/|W|
              loramcl  start LoRA-MCL (4 adapters), as wta1
            rft and rftbal share round 0. Bank: 4 slots per problem (modes / the 4 samples of a round), each keeps its
            latest correct solution with a \\boxed{} answer.
  rounds    gen(M_0) -> train M_1 -> gen(M_1) -> ... -> gen(M_{R-1}) -> train M_R -> gen(M_R): R acquisition rounds
            on the pool (4 attempts each); hv and test at M_0, every even round and M_R (hv: 16 attempts at M_0 and
            M_R, else 4; test: 4); at M_0 the frozen set gets 20 more attempts that never enter a bank (the
            discoveries a model would make by sampling alone, 6 x 4 attempts)
  swap      sftmulti trained on the final bank of wta1 (rftbal weights): better data or better use of it?
  report    phase1/sreport.py -> report_self.md; after seed 0 already, then again at the end
Seed s > 0 starts from the adapters of phase1_seeds of that seed; seed 0 has priority, so it finishes first.
Disk: one export (3.5 GB) per running generation, intermediate adapters are deleted after their generation: the peak
is ~16 GB."""
import argparse
import json
import os
import shutil
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase1.seeds_run import PY, log  # noqa: E402
from pilot.common import is_done  # noqa: E402
from pilot.run_all import Job, Runner, simulate  # noqa: E402

CHAINS = ("rftbal", "wta1", "loramcl", "rft")
START = {"rft": "sftmulti", "rftbal": "sftmulti", "wta1": "wta1", "loramcl": "loramcl", "swap": "sftmulti"}
KIND = {"rft": "rft", "rftbal": "rftbal", "wta1": "modes", "loramcl": "modes", "swap": "swap"}
UNITS = {"sftmulti": 1, "wta1": 4, "loramcl": 4}           # checkpoints per generation step
EXPORTS = {"sftmulti": 1, "wta1": 1, "loramcl": 4}          # full exports per generation step (FiLM: one shared)
ID = {"rft": 1, "rftbal": 2, "wta1": 3, "loramcl": 4, "swap": 5, "sftmulti": 6}
N_TEST = 1500


def start_adapter(a, s, st):
    if s == 0:
        return f"{a.seeds_run}/train/loramcl_s0/adapter.pt" if st == "loramcl" else f"{a.phase1}/train/{st}/adapter.pt"
    return f"{a.seeds_run}/train/{st}_s{s}/adapter.pt"


def eval_round(r, R):
    return r == 0 or r == R or r % 2 == 0


def sets(r, R):
    """Attempts per problem of every set at round r (r = R: the final model, no pool)."""
    out = {}
    if r < R:
        out["pool"] = 4
    if eval_round(r, R):
        out["hv"] = 16 if r in (0, R) else 4
        out["test"] = 4
    if r == 0:
        out["frozen"] = 20
    return out


def gen_seed(s, r, who):
    return 1_000_000 * (s + 1) + 10_000 * r + 100 * ID[who]       # mode m adds 1000 * (m + 1)


def sizes(a, run):
    m = os.path.join(run, "data", "meta.json")
    if os.path.exists(m):
        return json.load(open(m))["sizes"]
    return {"pool": a.n_hard + a.n_med, "hv": a.n_hv, "frozen": a.n_frozen, "test": N_TEST}


def est_gen(a, st, spec, N):
    sols = sum(n * N[k] for k, n in spec.items())
    return (sols / a.gen_per_min + UNITS[st] * a.start_min + EXPORTS[st] * a.export_min) / 60


def jobs(a, run, model):
    J = []
    data = f"{run}/data"
    p = ["--phase0", a.phase0, "--phase1", a.phase1, "--out", run, "--shards", str(a.shards)]
    sm = ["--n_math", "0", "--n_synth", "24", "--ops", "3", "5"] if a.smoke else \
        ["--n_math", str(a.n_math), "--n_synth", str(a.n_synth), "--ops", *map(str, a.ops)]
    J.append(Job("pool_cands", [PY, "-m", "phase1.spool", "cands", *p, *sm], f"{run}/pool", weight=0, est=0.1, prio=100))
    n_cand = (a.n_synth + (1300 if a.n_math else 0)) / a.shards
    for i in range(a.shards):
        J.append(Job(f"screen_{i}", [PY, "-m", "phase1.spool", "screen", "--out", run, "--model", model, "--shard",
                                     str(i), "--seed", "7", "--max_tokens", str(a.max_tokens)],
                     f"{run}/pool/screen/shard{i}", ["pool_cands"], weight=2,
                     est=(n_cand * 8 / a.gen_per_min + a.start_min) / 60, prio=100))
    sel = ["--n_hv", str(a.n_hv), "--n_hard", str(a.n_hard), "--n_med", str(a.n_med), "--n_frozen", str(a.n_frozen)]
    J.append(Job("pool_select", [PY, "-m", "phase1.spool", "select", *p, *sel], data,
                 [f"screen_{i}" for i in range(a.shards)], weight=0, est=0.02, prio=100))
    N = sizes(a, run)
    R = a.rounds
    tr = ["--steps_max", str(a.steps_max), "--epochs", str(a.epochs), "--P", str(a.P), "--lr", str(a.lr),
          "--film_lr_ratio", str(a.film_lr_ratio), "--max_hours", str(a.round_hours)]

    def gen_job(name, adapter, out, spec, seed, deps, st, prio, drop=0):
        J.append(Job(name, [PY, "-m", "phase1.selftrain", "gen", "--base", model, "--adapter", adapter, "--data", data,
                            "--out", out, "--sets", ",".join(f"{k}:{n}" for k, n in spec.items()), "--seed", str(seed),
                            "--max_tokens", str(a.max_tokens), "--drop_adapter", str(drop)],
                     out, deps, weight=2, est=est_gen(a, st, spec, N), prio=prio))

    def train_job(name, init, gdirs, kind, out, seed, deps, prio):
        J.append(Job(name, [PY, "-m", "phase1.selftrain", "train", "--base", model, "--init", init, "--gens",
                            ",".join(gdirs), "--kind", kind, "--data", data, "--out", out, "--seed", str(seed), *tr],
                     out, deps, weight=2, est=a.train_min / 60, prio=prio))

    for si, s in enumerate(a.seeds):
        prio = 10 * (len(a.seeds) - si)
        starts = sorted({START[c] for c in a.chains})
        for st in starts:
            gen_job(f"gen_s{s}_{st}_r0", start_adapter(a, s, st), f"{run}/s{s}/start_{st}/gen", sets(0, R),
                    gen_seed(s, 0, st), ["pool_select"], st, prio)
        chains = list(a.chains) + (["swap"] if a.swap and "wta1" in a.chains else [])
        for c in chains:
            st = START[c]
            src = "wta1" if c == "swap" else c          # swap: the bank of wta1, trained from sftmulti
            gdirs = [f"{run}/s{s}/start_{START[src]}/gen"] + [f"{run}/s{s}/{src}/r{r}/gen" for r in range(1, R)]
            dep = lambda r: f"gen_s{s}_{START[src]}_r0" if r == 0 else f"gen_s{s}_{src}_r{r}"
            if c == "swap":
                out = f"{run}/s{s}/swap"
                train_job(f"train_s{s}_swap", start_adapter(a, s, st), gdirs, "swap", f"{out}/train", 100 * s + R,
                          [dep(R - 1)], prio)
                gen_job(f"gen_s{s}_swap", f"{out}/train/adapter.pt", f"{out}/gen", sets(R, R), gen_seed(s, R, "swap"),
                        [f"train_s{s}_swap"], st, prio)
                continue
            for r in range(1, R + 1):
                d = f"{run}/s{s}/{c}/r{r}"
                train_job(f"train_s{s}_{c}_r{r}", start_adapter(a, s, st), gdirs[:r], KIND[c], f"{d}/train",
                          100 * s + r, [dep(r - 1)], prio)
                gen_job(f"gen_s{s}_{c}_r{r}", f"{d}/train/adapter.pt", f"{d}/gen", sets(r, R), gen_seed(s, r, c),
                        [f"train_s{s}_{c}_r{r}"], st, prio, drop=int(r < R))
    rep = [PY, "-m", "phase1.sreport", "--run", run, "--seeds", ",".join(map(str, a.seeds)), "--chains",
           ",".join(a.chains), "--rounds", str(R)] + (["--boot", "20"] if a.smoke else [])
    gens = [j.name for j in J if j.name.startswith("gen_s")]
    if len(a.seeds) > 1:
        s0 = [g for g in gens if g.startswith(f"gen_s{a.seeds[0]}_")]
        J.append(Job(f"report_s{a.seeds[0]}", rep + ["--mark", f"{run}/report_job_s{a.seeds[0]}"],
                     f"{run}/report_job_s{a.seeds[0]}", s0, weight=0, est=0.05, soft=True))
    J.append(Job("report", rep + ["--mark", f"{run}/report_job"], f"{run}/report_job", gens, weight=0, est=0.05,
                 soft=True))
    return J


def check_inputs(a):
    need = [f"{a.phase0}/data/pool.jsonl", f"{a.phase0}/pool", f"{a.phase1}/hard/pool.jsonl", f"{a.phase1}/hard/out"]
    need += [f"{a.phase1}/data/{f}.jsonl" for f in ("train", "test")]
    chains = list(a.chains) + (["swap"] if a.swap and "wta1" in a.chains else [])
    need += sorted({start_adapter(a, s, START[c]) for s in a.seeds for c in chains})
    return [p for p in need if not os.path.exists(p)]


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/phase1_self")
    ap.add_argument("--phase0", default="runs/phase0")
    ap.add_argument("--phase1", default="runs/phase1")
    ap.add_argument("--seeds_run", default="runs/phase1_seeds", help="loramcl of seed 0, all start adapters of seed > 0")
    ap.add_argument("--model", default="")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--seeds", default="0,1", help="seed 0 first; a seed that would not finish in time is cut")
    ap.add_argument("--chains", default=",".join(CHAINS))
    ap.add_argument("--swap", type=int, default=1, help="1: sftmulti trained on the final bank of wta1")
    ap.add_argument("--rounds", type=int, default=6, help="acquisition rounds R")
    ap.add_argument("--hours", type=float, default=14.0, help="planned time; only a warning if the plan exceeds it")
    ap.add_argument("--max_hours", type=float, default=16.0,
                    help="from the start of this invocation: jobs that would end later are skipped")
    ap.add_argument("--max_tokens", type=int, default=1200)
    ap.add_argument("--shards", type=int, default=4, help="screening jobs")
    ap.add_argument("--n_math", type=int, default=-1, help="fresh MATH candidates (-1: all)")
    ap.add_argument("--n_synth", type=int, default=5000, help="fresh synthetic candidates")
    ap.add_argument("--ops", type=int, nargs=2, default=(8, 13), help="operations of the synthetic candidates")
    ap.add_argument("--n_hv", type=int, default=800)
    ap.add_argument("--n_hard", type=int, default=1800)
    ap.add_argument("--n_med", type=int, default=600)
    ap.add_argument("--n_frozen", type=int, default=600)
    ap.add_argument("--steps_max", type=int, default=300, help="optimizer steps per round at most")
    ap.add_argument("--epochs", type=float, default=2.0, help="passes over the bank per round at most")
    ap.add_argument("--P", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--film_lr_ratio", type=float, default=5.0, help="FiLM LR / LoRA LR (phase 1: 1e-3 / 2e-4)")
    ap.add_argument("--round_hours", type=float, default=2.0,
                    help="safety time limit of one training round, far above a normal one (the report flags it)")
    ap.add_argument("--gen_per_min", type=float, default=850, help="estimate: solutions per minute per GPU")
    ap.add_argument("--start_min", type=float, default=1.0, help="estimate: vLLM start per checkpoint, minutes")
    ap.add_argument("--export_min", type=float, default=2.0, help="estimate: one full export, minutes")
    ap.add_argument("--train_min", type=float, default=15.0, help="estimate: one training round, minutes")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--plan_only", action="store_true", help="print the jobs and the simulated time, run nothing")
    a = ap.parse_args(argv)
    a.seeds = [int(s) for s in a.seeds.split(",")]
    a.chains = [c for c in a.chains.split(",") if c]
    if any(c not in CHAINS for c in a.chains) or len(set(a.seeds)) != len(a.seeds):
        raise SystemExit(f"--chains from {CHAINS}, distinct --seeds")
    a.phase0, a.phase1, a.seeds_run = map(os.path.abspath, (a.phase0, a.phase1, a.seeds_run))
    if a.smoke:                  # every kind of job once: the first seed, 2 rounds, tiny sets
        a.seeds = a.seeds[:1]
        a.max_tokens, a.rounds, a.steps_max, a.P, a.shards = min(a.max_tokens, 384), 2, 2, 4, 2
        a.n_hv, a.n_hard, a.n_med, a.n_frozen, a.round_hours = 4, 8, 4, 3, 0.2
    if a.rounds < 1:
        raise SystemExit("--rounds must be >= 1")
    return a


def main():
    a = parse()
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
    # finished steps are reused on resume: settings that change their results must stay the same (seeds and chains
    # may be added)
    fixed = {"model_path": os.path.abspath(model), "max_tokens": a.max_tokens, "rounds": a.rounds,
             "steps_max": a.steps_max, "epochs": a.epochs, "P": a.P, "lr": a.lr, "round_hours": a.round_hours,
             "film_lr_ratio": a.film_lr_ratio,
             "smoke": a.smoke, "phase0": a.phase0, "phase1": a.phase1, "seeds_run": a.seeds_run,
             "pool": [a.n_math, a.n_synth, list(a.ops), a.n_hv, a.n_hard, a.n_med, a.n_frozen, a.shards]}
    for k, v in fixed.items():
        if state.setdefault(k, v) != v:
            raise SystemExit(f"{run} was started with {k} = {state[k]}, now {v}; use another --run")
    # the start adapters and the inputs of the pool: replaced files would mix two experiments in one directory
    chains = list(a.chains) + (["swap"] if a.swap and "wta1" in a.chains else [])
    inputs = sorted({start_adapter(a, s, START[c]) for s in a.seeds for c in chains} |
                    {f"{a.phase1}/data/{f}.jsonl" for f in ("train", "test", "verify")} |
                    {f"{a.phase0}/data/pool.jsonl", f"{a.phase1}/hard/pool.jsonl"})
    fp = state.setdefault("inputs", {})
    for f in inputs:
        if os.path.exists(f):
            v = [os.path.getsize(f), int(os.path.getmtime(f))]
            if fp.setdefault(f, v) != v:
                raise SystemExit(f"{f} changed since this run started (size, mtime {fp[f]} -> {v}); use another --run")
    state["seeds"] = sorted(set(state.get("seeds", [])) | set(a.seeds))
    state["chains"] = [c for c in CHAINS if c in set(state.get("chains", [])) | set(a.chains)]
    J = jobs(a, run, model)
    for j in J:
        if is_done(j.out):
            j.est = 0.0
    mk = simulate(J, gpus)
    s0 = simulate([j for j in J if not j.name.startswith("report") and (
        j.prio >= 10 * len(a.seeds) or j.prio == 100)], gpus)
    log(f"{len(J)} jobs, simulated {mk:.1f} h left on {len(gpus)} GPUs (seed {a.seeds[0]} alone {s0:.1f} h; budget "
        f"{a.hours} h, hard limit {a.max_hours} h); seeds {a.seeds}, chains {a.chains}, {a.rounds} rounds; "
        f"free disk {shutil.disk_usage(run).free / 1e9:.0f} GB (as the OS reports it)")
    if a.plan_only:
        for j in J:
            log(f"  {j.name}: est {j.est:.2f} h, prio {j.prio}, deps {j.deps}")
        return
    if mk > a.hours:
        log(f"WARNING: the plan ({mk:.1f} h) is longer than the budget ({a.hours} h): the last seed may be cut")
    session = time.time()
    state.setdefault("started", session)
    state.update(session_started=session, planned_hours=mk)
    json.dump(state, open(sp, "w"), indent=1)
    for d in os.listdir(run):              # the reports cover what exists now
        if d.startswith("report_job"):
            shutil.rmtree(os.path.join(run, d), ignore_errors=True)
    runner = Runner(run, gpus, state, disk_limit_gb=1e9)

    def on_signal(sig, frame):
        log("signal received -- stopping children")
        runner.kill_all()
        sys.exit(1)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    ok = runner.run(J, "phase1-self-smoke" if a.smoke else "phase1-self", deadline=session + a.max_hours * 3600)
    rp = os.path.join(run, "report_self.md")
    log(f"finished; report: {rp}" if os.path.exists(rp) else "finished without report -- see logs/")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
