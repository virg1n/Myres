"""Phase 1 with one command (all GPU work scheduled over the available GPUs, resumable).

  hard_prep -> hard_gen x4 (harder problems, 8 solutions each) -> build (refs, verify, val, stage-2 set)
  -> train x6 variants -> eval x6 (val + test) -> select (on val) -> stage 2 (best + SFT-multi control)
  -> eval x2 -> report;   noise2 (larger-sigma noise baseline) fills idle GPUs.
Training jobs have time limits; if the simulated schedule does not fit --hours they are scaled down."""
import argparse
import json
import os
import shlex
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pilot.run_all import Job, Runner, simulate  # noqa: E402

PY = sys.executable
# equal wall-clock budget for every variant (roughly equal compute); cheaper ones make several passes
TRAIN_H = {"wtamulti": 2.0, "wtabal": 2.0, "wta1": 2.0, "filmmean": 2.0, "sftmulti": 2.0, "sft1": 2.0}
MODE_V = {"wtamulti", "wtabal", "wta1", "filmmean"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def marked(cmd, mark):
    return ["bash", "-c", f"{shlex.join(cmd)} && mkdir -p {shlex.quote(mark)} && touch {shlex.quote(mark)}/DONE"]


def jobs(a, run, model, scale):
    p0 = a.phase0
    sm = a.smoke
    J = []
    hard_args = (["--n_math", "16", "--n_synth", "16", "--max_resample", "16"] if sm else []) + \
        ([] if a.hard_math < 0 else ["--n_math", str(a.hard_math)])
    J.append(Job("hard_prep", [PY, "-m", "phase1.hard", "--phase0", p0, "--out", run] + hard_args,
                 f"{run}/hard", weight=0, est=0.05))
    mt = ["--max_tokens", str(a.max_tokens)]
    for s in range(a.gen_shards):
        cmd = [PY, "-m", "phase0.gen", "pool", "--model", model, "--data", f"{run}/hard", "--out", f"{run}/hard/out",
               "--shard", str(s), "--nshards", str(a.gen_shards), "--n", "2"] + mt
        J.append(Job(f"hard_gen_{s}", marked(cmd, f"{run}/hard/done_{s}"), f"{run}/hard/done_{s}", ["hard_prep"],
                     weight=2, est=0.5))
    bargs = ["--n_val", "12", "--n_verify", "24", "--n_stage2", "12", "--max_train", "24", "--max_test", "16"] if sm else []
    J.append(Job("build", [PY, "-m", "phase1.build", "--phase0", p0, "--out", run] + bargs, f"{run}/data",
                 [f"hard_gen_{s}" for s in range(a.gen_shards)], weight=0, est=0.1))
    targs = ["--max_steps", "2", "--P", "4"] if sm else []
    for v, h in TRAIN_H.items():
        cap = max(0.1, h * scale)
        J.append(Job(f"train_{v}", [PY, "-m", "phase1.train", "--variant", v, "--base", model, "--data", f"{run}/data",
                                    "--out", f"{run}/train/{v}", "--max_hours", f"{cap:.2f}"] + targs,
                     f"{run}/train/{v}", ["build"], weight=2, est=cap + 0.1))
    ev_modes = "2" if sm else "4"
    eval_vs = ["sftmulti", "wtabal"] if sm else list(TRAIN_H)
    for v in eval_vs:
        J.append(Job(f"eval_{v}", [PY, "-m", "phase1.evaluate", "--base", model, "--adapter", f"{run}/train/{v}/adapter.pt",
                                   "--data", f"{run}/data", "--out", f"{run}/eval/{v}", "--modes", ev_modes] + mt,
                     f"{run}/eval/{v}", [f"train_{v}"], weight=2, est=1.0 if v in MODE_V else 0.6))
    J.append(Job("select", [PY, "-m", "phase1.select", "--run", run], f"{run}/select",
                 [f"eval_{v}" for v in eval_vs], weight=0, est=0.05))
    s2 = ["--rounds", "1", "--max_steps", "2"] if sm else ["--round_hours", f"{max(0.15, 0.5 * scale):.2f}"]
    J.append(Job("stage2_best", [PY, "-m", "phase1.stage2", "--base", model, "--choice", f"{run}/select/choice.json",
                                 "--data", f"{run}/data", "--out", f"{run}/stage2/s2_best"] + s2 + mt,
                 f"{run}/stage2/s2_best", ["select"], weight=2, est=2.0 * max(scale, 0.5)))
    J.append(Job("stage2_ctrl", [PY, "-m", "phase1.stage2", "--base", model,
                                 "--adapter", f"{run}/train/sftmulti/adapter.pt", "--data", f"{run}/data",
                                 "--out", f"{run}/stage2/s2_ctrl"] + s2 + mt,
                 f"{run}/stage2/s2_ctrl", ["train_sftmulti", "build"], weight=2, est=1.5 * max(scale, 0.5)))
    for name, ests in (("s2_best", 1.0), ("s2_ctrl", 0.6)):
        J.append(Job(f"eval_{name}", [PY, "-m", "phase1.evaluate", "--base", model,
                                      "--adapter", f"{run}/stage2/{name}/adapter.pt", "--data", f"{run}/data",
                                      "--out", f"{run}/eval/{name}", "--modes", ev_modes, "--sets", "test"] + mt,
                     f"{run}/eval/{name}", [f"stage2_{name[3:]}"], weight=2, est=ests))
    if not sm:
        J.append(Job("noise2", [PY, "-m", "phase1.noise2", "--phase0", p0, "--run", f"{run}/noise2", "--model", model]
                     + mt, f"{run}/noise2", ["build"], weight=2, est=1.2))
    rep = [PY, "-m", "phase1.report", "--run", run, "--phase0", "" if sm else p0, "--noise2", f"{run}/noise2",
           "--mark", f"{run}/report_job"]
    J.append(Job("report", rep, f"{run}/report_job", [j.name for j in J if j.name.startswith(("eval_", "noise2"))],
                 weight=0, est=0.05, soft=True))
    return J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/phase1")
    ap.add_argument("--phase0", default="runs/phase0")
    ap.add_argument("--model", default="")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--hours", type=float, default=12.0)
    ap.add_argument("--gen_shards", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=1200)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--hard_math", type=int, default=-1, help="MATH level-5 problems to add (-1 = default)")
    a = ap.parse_args()
    a.phase0 = os.path.abspath(a.phase0)
    if a.smoke:
        a.max_tokens, a.gen_shards = min(a.max_tokens, 384), 2
    run = os.path.abspath(a.run)
    os.makedirs(run, exist_ok=True)
    model = a.model
    if not model:
        st0 = os.path.join(a.phase0, "state.json")
        model = json.load(open(st0)).get("model_path") if os.path.exists(st0) else ""
    if not model or not os.path.isdir(model):
        from huggingface_hub import snapshot_download
        model = snapshot_download(a.model or "Qwen/Qwen3-1.7B",
                                  allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*", "merges*", "vocab*"])
    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    sp = os.path.join(run, "state.json")
    state = json.load(open(sp)) if os.path.exists(sp) else {}
    if "scale" not in state:
        scale = 1.0
        for _ in range(12):
            mk = simulate(jobs(a, run, model, scale), gpus)
            if mk <= a.hours * 0.9 or a.smoke:
                break
            scale *= 0.85
        state.update(scale=scale, planned_hours=mk, started=time.time(), model_path=model)
        json.dump(state, open(sp, "w"), indent=1)
        log(f"plan: simulated {mk:.1f} h for budget {a.hours} h (training time scale {scale:.2f})")
    J = jobs(a, run, model, state["scale"])
    ok = Runner(run, gpus, state, disk_limit_gb=1e9).run(J, "phase1-smoke" if a.smoke else "phase1")
    rp = os.path.join(run, "report_phase1.md")
    log(f"finished; report: {rp}" if os.path.exists(rp) else "finished without report -- see logs/")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
