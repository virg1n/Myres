"""The whole pilot with one command.

  preflight   tiny versions of every job type (catches bugs before the night)
  calibration 4 short baselines with different op_max -> pick difficulty; measure throughput
  plan        largest token budget T whose simulated schedule fits into --hours
  main        stage 1 (6 runs) -> natural errors -> stage 2 (4 runs) -> 12 evaluations -> report

Re-running the same command resumes: finished jobs (DONE marker) are skipped, training resumes
from last.pt. Data is generated on the fly, so disk use is only checkpoints/logs (a few GB).
"""
import argparse
import os
import signal
import subprocess
import sys
import time

from .common import disk_usage_gb, free_gb, is_done, log, read_json, write_json

PY = sys.executable
GPU_CAPACITY = 2          # a training job takes the whole GPU (weight 2), generation jobs share (weight 1)


class Job:
    def __init__(self, name, cmd, out, deps=(), weight=1, est=0.1, soft=False):
        self.name, self.cmd, self.out, self.deps = name, cmd, out, list(deps)
        self.weight, self.est, self.soft = weight, est, soft
        self.status, self.gpu, self.proc, self.t0, self.t1, self.logf = "pending", None, None, None, None, None


# ----------------------------------------------------------------------------- scheduling
def ranks(jobs):
    """Upward rank = est + max rank of dependents (critical-path priority)."""
    by = {j.name: j for j in jobs}
    users = {j.name: [] for j in jobs}
    for j in jobs:
        for d in j.deps:
            if d in users:
                users[d].append(j.name)
    memo = {}

    def r(n):
        if n not in memo:
            memo[n] = by[n].est + max((r(u) for u in users[n]), default=0.0)
        return memo[n]

    return {j.name: r(j.name) for j in jobs}


def pick_gpu(load, weight):
    """GPU with enough free capacity; generation jobs are packed onto already shared GPUs."""
    fits = [g for g, l in load.items() if GPU_CAPACITY - l >= weight]
    if not fits:
        return None
    return max(fits, key=lambda g: (load[g], -int(g))) if weight == 1 else min(fits, key=lambda g: (load[g], int(g)))


def simulate(jobs, gpus):
    """Makespan (hours) of the same list-scheduling policy the runner uses."""
    rk = ranks(jobs)
    done, running, t = set(), [], 0.0
    load = {g: 0 for g in gpus}
    pending = list(jobs)
    while pending or running:
        ready = sorted((j for j in pending if all(d in done for d in j.deps)), key=lambda j: -rk[j.name])
        for j in ready:
            if j.weight == 0:
                running.append((t + j.est, j, None))
                pending.remove(j)
                continue
            g = pick_gpu(load, j.weight)
            if g is None:
                continue
            load[g] += j.weight
            running.append((t + j.est, j, g))
            pending.remove(j)
        if not running:
            break
        running.sort(key=lambda x: x[0])
        t, j, g = running.pop(0)
        done.add(j.name)
        if g is not None:
            load[g] -= j.weight
    return t


class Runner:
    def __init__(self, run_dir, gpus, state, disk_limit_gb):
        self.run_dir, self.gpus, self.state, self.disk_limit = run_dir, gpus, state, disk_limit_gb
        self.children = []
        os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)

    def run(self, jobs, title):
        rk = ranks(jobs)
        load = {g: 0 for g in self.gpus}
        by = {j.name: j for j in jobs}
        for j in jobs:
            if is_done(j.out):
                j.status = "done"
        last_status = 0
        while True:
            for j in jobs:
                if j.status != "pending":
                    continue
                st = [by[d].status if d in by else "done" for d in j.deps]
                if not j.soft and any(s in ("failed", "skipped") for s in st):
                    j.status = "skipped"
            ready = sorted((j for j in jobs if j.status == "pending" and all(
                (by[d].status in ("done", "failed", "skipped")) if j.soft else (by[d].status == "done")
                for d in j.deps if d in by)), key=lambda j: -rk[j.name])
            for j in ready:
                if not self._disk_ok():
                    break
                if j.weight == 0:
                    self._start(j, None)
                    continue
                g = pick_gpu(load, j.weight)
                if g is None:
                    continue
                load[g] += j.weight
                self._start(j, g)
            for j in jobs:
                if j.status == "running" and j.proc.poll() is not None:
                    j.t1 = time.time()
                    j.logf.close()
                    ok = j.proc.returncode == 0 and is_done(j.out)
                    j.status = "done" if ok else "failed"
                    if j.gpu is not None:
                        load[j.gpu] -= j.weight
                    log(f"[{title}] {j.name}: {j.status} after {(j.t1 - j.t0) / 3600:.2f}h"
                        + ("" if ok else f" (see logs/{j.name}.log)"))
            if time.time() - last_status > 20:
                self._status(jobs, title)
                last_status = time.time()
            if all(j.status in ("done", "failed", "skipped") for j in jobs):
                self._status(jobs, title)
                return all(j.status == "done" for j in jobs)
            if not any(j.status == "running" for j in jobs) and not ready:
                if not self._disk_ok():
                    log("disk limit reached and nothing is running -- stopping")
                    return False
            time.sleep(5)

    def _start(self, j, g):
        os.makedirs(j.out, exist_ok=True)
        env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        env["CUDA_VISIBLE_DEVICES"] = "" if g is None else str(g)
        j.logf = open(os.path.join(self.run_dir, "logs", f"{j.name}.log"), "a")
        j.logf.write(f"\n### {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(j.cmd)}\n")
        j.logf.flush()
        j.proc = subprocess.Popen(j.cmd, stdout=j.logf, stderr=subprocess.STDOUT, env=env,
                                  start_new_session=True)
        self.children.append(j.proc)
        j.status, j.gpu, j.t0 = "running", g, time.time()
        log(f"start {j.name} on {'cpu' if g is None else 'gpu ' + str(g)} (est {j.est:.2f}h)")

    def _disk_ok(self):
        used = disk_usage_gb(self.run_dir)
        if used > self.disk_limit - 2 or free_gb(self.run_dir) < 2:
            log(f"WARNING: disk {used:.1f} GB used in run dir (limit {self.disk_limit}) -- not starting new jobs")
            return False
        return True

    def _status(self, jobs, title):
        jst = self.state.setdefault("jobs", {})
        for j in jobs:
            jst[j.name] = {"status": j.status, "gpu": j.gpu, "est_h": round(j.est, 2),
                           "hours": round(((j.t1 or time.time()) - j.t0) / 3600, 2) if j.t0 else None}
        write_json(os.path.join(self.run_dir, "state.json"), self.state)
        lines = [f"# Статус пилота — {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
                 f"Фаза: **{title}**. Диск: {disk_usage_gb(self.run_dir):.1f} GB.\n"]
        plan = self.state.get("plan")
        if plan:
            lines.append(f"План: op_max={plan['op_max']}, T={plan['tokens'] / 1e9:.2f}B токенов, "
                         f"ожидаемое окончание ≈ {plan.get('eta', '?')}\n")
        lines.append("| задача | статус | GPU | часов | оценка, ч | последняя строка лога |")
        lines.append("|---|---|---|---|---|---|")
        for j in jobs:
            tail = ""
            if j.status in ("running", "failed"):
                tail = last_line(os.path.join(self.run_dir, "logs", f"{j.name}.log"))
            h = f"{((j.t1 or time.time()) - j.t0) / 3600:.2f}" if j.t0 else ""
            lines.append(f"| {j.name} | {j.status} | {'' if j.gpu is None else j.gpu} | {h} | {j.est:.2f} | "
                         f"{tail.replace('|', '/')[:150]} |")
        with open(os.path.join(self.run_dir, "STATUS.md"), "w") as f:
            f.write("\n".join(lines) + "\n")

    def kill_all(self):
        for p in self.children:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except Exception:
                    pass


def last_line(path):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 2000))
            lines = [l for l in f.read().decode(errors="ignore").splitlines() if l.strip()]
            return lines[-1] if lines else ""
    except OSError:
        return ""


# ----------------------------------------------------------------------------- job graphs
def train_cmd(out, method, z, tokens, op_max, size, workers, extra=()):
    return [PY, "-m", "pilot.train", "--out", out, "--method", method, "--z", z, "--tokens", str(int(tokens)),
            "--op_max", str(op_max), "--size", size, "--workers", str(workers), "--seed", "1", *extra]


def preflight_jobs(d, workers):
    t = lambda m, z, extra=(): train_cmd(f"{d}/train/{m}_{z}", m, z, 2e5, 6, "tiny", min(workers, 2),
                                         ["--micro_bs", "16", "--accum", "1", "--quick_eval_n", "16",
                                          "--log_every", "5", *extra])
    ev = lambda ck, name, extra=(): [PY, "-m", "pilot.evaluate", "--ckpt", ck, "--out", f"{d}/eval/{name}",
                                     "--n_test", "16", "--n_val", "8", *extra]
    jobs = [
        Job("pf_train_base", t("base", "none", ["--early_frac", "0.3"]), f"{d}/train/base_none", weight=2),
        Job("pf_train_wta_fix", t("wta", "fix"), f"{d}/train/wta_fix", weight=2),
        Job("pf_train_wta_cont", t("wta", "cont"), f"{d}/train/wta_cont", weight=2),
        Job("pf_train_mean_cont", t("mean", "cont"), f"{d}/train/mean_cont", weight=2),
        Job("pf_bench", train_cmd(f"{d}/bench", "wta", "fix", 1e12, 6, "tiny", min(workers, 2),
                                  ["--micro_bs", "16", "--accum", "1", "--bench_steps", "5"]), f"{d}/bench", weight=2),
        Job("pf_collect", [PY, "-m", "pilot.collect", "--out", f"{d}/collect", "--n", "24", "--ckpts",
                           f"{d}/train/base_none/final.pt", f"{d}/train/base_none/early.pt"],
            f"{d}/collect", ["pf_train_base"]),
        Job("pf_s2_fix", [PY, "-m", "pilot.stage2", "--init", f"{d}/train/wta_fix/final.pt", "--out",
                          f"{d}/stage2/fix", "--natural", f"{d}/collect/natural.jsonl", "--problems", "32",
                          "--P", "16", "--micro_bs", "16", "--log_every", "1"],
            f"{d}/stage2/fix", ["pf_train_wta_fix", "pf_collect"]),
        Job("pf_s2_base", [PY, "-m", "pilot.stage2", "--init", f"{d}/train/base_none/final.pt", "--out",
                           f"{d}/stage2/base", "--natural", f"{d}/collect/natural.jsonl", "--problems", "16",
                           "--P", "16", "--weights", "uniform", "--micro_bs", "16"],
            f"{d}/stage2/base", ["pf_train_base", "pf_collect"]),
        Job("pf_eval_A", ev(f"{d}/train/base_none/final.pt", "A"), f"{d}/eval/A", ["pf_train_base"]),
        Job("pf_eval_C_fix2", ev(f"{d}/stage2/fix/final.pt", "C_fix2"), f"{d}/eval/C_fix2", ["pf_s2_fix"]),
        Job("pf_eval_A_RFT", ev(f"{d}/stage2/base/final.pt", "A_RFT"), f"{d}/eval/A_RFT", ["pf_s2_base"]),
        Job("pf_eval_C_cont", ev(f"{d}/train/wta_cont/final.pt", "C_cont"), f"{d}/eval/C_cont",
            ["pf_train_wta_cont"]),
        Job("pf_eval_B_cont", ev(f"{d}/train/base_none/final.pt", "B_cont", ["--noise", "cont", "--sigmas", "0.02"]),
            f"{d}/eval/B_cont", ["pf_train_base"]),
    ]
    jobs.append(Job("pf_report", [PY, "-m", "pilot.report", "--run", d, "--mark", f"{d}/report_job"],
                    f"{d}/report_job", [j.name for j in jobs if j.name.startswith("pf_eval")], weight=0, soft=True))
    return jobs


def main_jobs(d, p, size, workers, n_test, n_val, s2_problems, n_nat):
    T, op, K = p["tokens"], p["op_max"], 4
    r_flops = 1 + K / 3
    hb, hw, gps = p["tps_base"] * 3600, p["tps_wta"] * 3600, p["gps"] * 3600
    gen_slow = 1.3        # two generation jobs share a GPU
    ev_gens = n_val * 4 * 4 + 2 * n_test * 16
    J = []

    def tr(name, method, z, tokens, est, extra=()):
        J.append(Job(f"train_{name}", train_cmd(f"{d}/train/{name}", method, z, tokens, op, size, workers, extra),
                     f"{d}/train/{name}", weight=2, est=est + 0.1))

    tr("A", "base", "none", T, T / hb, ["--early_frac", "0.33"])
    tr("Aprime", "base", "none", T * r_flops, T * r_flops / hb)
    tr("C_fix", "wta", "fix", T, T / hw)
    tr("C_cont", "wta", "cont", T, T / hw)
    tr("D_fix", "mean", "fix", T, 1.05 * T / hb)
    tr("D_cont", "mean", "cont", T, 1.05 * T / hb)
    J.append(Job("collect", [PY, "-m", "pilot.collect", "--out", f"{d}/collect", "--n", str(n_nat), "--ckpts",
                             f"{d}/train/A/final.pt", f"{d}/train/A/early.pt"], f"{d}/collect", ["train_A"],
                 est=4 * n_nat / gps * gen_slow + 0.05))
    s2_est = s2_problems * K / gps * 1.3 * gen_slow + 0.1
    for name, init, w in (("C_fix2", "C_fix", "unique"), ("C_fix2u", "C_fix", "uniform"),
                          ("C_cont2", "C_cont", "unique"), ("A_RFT", "A", "uniform")):
        J.append(Job(f"stage2_{name}", [PY, "-m", "pilot.stage2", "--init", f"{d}/train/{init}/final.pt",
                                        "--out", f"{d}/stage2/{name}", "--natural", f"{d}/collect/natural.jsonl",
                                        "--weights", w, "--problems", str(s2_problems)],
                     f"{d}/stage2/{name}", [f"train_{init}", "collect"], est=s2_est))
    evals = {"A": ("train/A", ""), "Aprime": ("train/Aprime", ""), "B_fix": ("train/A", "fix"),
             "B_cont": ("train/A", "cont"), "C_fix": ("train/C_fix", ""), "C_cont": ("train/C_cont", ""),
             "D_fix": ("train/D_fix", ""), "D_cont": ("train/D_cont", ""), "A_RFT": ("stage2/A_RFT", ""),
             "C_fix2": ("stage2/C_fix2", ""), "C_fix2u": ("stage2/C_fix2u", ""), "C_cont2": ("stage2/C_cont2", "")}
    for name, (src, noise) in evals.items():
        dep = ("train_" if src.startswith("train") else "stage2_") + src.split("/")[1]
        gens = ev_gens + (2 * n_test * 16 if "cont" in name and not noise else 0) + \
            (4 * 2 * 4 * n_val if noise else 0)
        cmd = [PY, "-m", "pilot.evaluate", "--ckpt", f"{d}/{src}/final.pt", "--out", f"{d}/eval/{name}",
               "--n_test", str(n_test), "--n_val", str(n_val)] + (["--noise", noise] if noise else [])
        J.append(Job(f"eval_{name}", cmd, f"{d}/eval/{name}", [dep], est=gens / gps * 1.2 * gen_slow + 0.05))
    J.append(Job("report", [PY, "-m", "pilot.report", "--run", d, "--mark", f"{d}/report_job"],
                 f"{d}/report_job", [j.name for j in J if j.name.startswith("eval_")], weight=0, soft=True,
                 est=0.05))
    return J


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/pilot")
    ap.add_argument("--hours", type=float, default=11.0, help="total wall-clock budget from start")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--phase", choices=["preflight", "main", "all"], default="all")
    ap.add_argument("--size", default="20m")
    ap.add_argument("--calib_ops", type=int, nargs="+", default=[6, 9, 12, 15])
    ap.add_argument("--calib_tokens", type=float, default=2.5e8)
    ap.add_argument("--calib_target", type=float, default=0.5,
                    help="target greedy val accuracy of the short calibration run")
    ap.add_argument("--op_max", type=int, default=0, help="skip calibration of difficulty")
    ap.add_argument("--tokens", type=float, default=0, help="force stage-1 token budget T")
    ap.add_argument("--t_min", type=float, default=5e8)
    ap.add_argument("--t_max", type=float, default=3e9)
    ap.add_argument("--stage2_problems", type=int, default=60000)
    ap.add_argument("--n_natural", type=int, default=20000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--n_val", type=int, default=400)
    ap.add_argument("--disk_limit_gb", type=float, default=38.0)
    a = ap.parse_args()
    run = os.path.abspath(a.out)
    os.makedirs(run, exist_ok=True)
    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    workers = max(2, (os.cpu_count() or 8) // max(len(gpus), 1) - 2)
    sp = os.path.join(run, "state.json")
    state = read_json(sp) if os.path.exists(sp) else {}
    state.setdefault("started", time.time())
    deadline = state["started"] + a.hours * 3600
    runner = Runner(run, gpus, state, a.disk_limit_gb)

    def on_signal(sig, frame):
        log("signal received -- stopping children")
        runner.kill_all()
        sys.exit(1)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    if a.phase in ("preflight", "all") and not state.get("preflight_ok"):
        log("=== preflight: tiny run of every job type")
        d = os.path.join(run, "preflight")
        ok = runner.run(preflight_jobs(d, workers), "preflight")
        if not ok:
            log(f"PREFLIGHT FAILED -- see {run}/logs/pf_*.log and {run}/STATUS.md")
            sys.exit(1)
        state["preflight_ok"] = True
        write_json(sp, state)
        log(f"PREFLIGHT OK (report: {d}/report.md)")
        if a.phase == "preflight":
            return

    if "plan" not in state:
        log("=== calibration")
        cal = []
        ops = [a.op_max] if a.op_max else a.calib_ops
        for v in ops:
            cal.append(Job(f"calib_op{v}", train_cmd(f"{run}/calib/op{v}", "base", "none", a.calib_tokens, v, a.size,
                                                     workers, ["--quick_eval_n", "400"]),
                           f"{run}/calib/op{v}", weight=2, est=0.5))
        bench_op = ops[len(ops) // 2]
        cal.append(Job("bench_wta", train_cmd(f"{run}/calib/bench_wta", "wta", "fix", 1e12, bench_op, a.size, workers,
                                              ["--bench_steps", "40"]), f"{run}/calib/bench_wta", weight=2, est=0.1))
        if not runner.run(cal, "calibration"):
            log("calibration failed -- see logs/calib_*.log")
            sys.exit(1)
        res = {v: read_json(f"{run}/calib/op{v}/result.json") for v in ops}
        accs = {v: r["val_acc"] for v, r in res.items()}
        op_max = a.op_max or min(ops, key=lambda v: (abs(accs[v] - a.calib_target), -v))
        tps_wta = read_json(f"{run}/calib/bench_wta/bench.json")["tok_per_s"]
        tps = sorted(r["tok_per_s"] for r in res.values() if r.get("tok_per_s"))
        tps_base = tps[len(tps) // 2] if tps else tps_wta * (1 + 4 / 3)
        gps = res[op_max]["gen_per_s"]
        log(f"calibration acc by op_max: {accs} -> op_max={op_max}; tok/s base {tps_base:.0f} wta {tps_wta:.0f}; "
            f"gen/s {gps:.0f}")
        if max(accs.values()) < 0.02:
            w = (f"калибровка: точность почти 0 при всех op_max ({accs}) — модель за {a.calib_tokens / 1e9:.2f}B "
                 f"токенов не научилась; продолжаю с op_max={op_max}, но проверьте logs/calib_*.log")
            state.setdefault("warnings", []).append(w)
            log("WARNING:", w)
        p = {"op_max": op_max, "calib_acc": accs, "tps_base": tps_base, "tps_wta": tps_wta, "gps": gps,
             "stage2_problems": a.stage2_problems}
        left = (deadline - time.time()) / 3600 - 0.3
        mk = lambda T: simulate(main_jobs(run, dict(p, tokens=T), a.size, workers, a.n_test, a.n_val,
                                          a.stage2_problems, a.n_natural), gpus)
        if a.tokens:
            T = a.tokens
        else:
            lo, hi = a.t_min, a.t_max
            if mk(hi) <= left * 0.92:
                T = hi
            elif mk(lo) > left * 0.92:
                T = lo
                log(f"WARNING: even T={lo / 1e9:.2f}B needs {mk(lo):.1f}h > {left:.1f}h left; results will be late")
            else:
                for _ in range(30):
                    mid = (lo + hi) / 2
                    lo, hi = (mid, hi) if mk(mid) <= left * 0.92 else (lo, mid)
                T = lo
        p["tokens"] = int(T)
        p["tokens_aprime"] = int(T * (1 + 4 / 3))
        p["makespan_h"] = mk(T)
        p["eta"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + p["makespan_h"] * 3600))
        state["plan"] = p
        write_json(sp, state)
        log(f"plan: T={T / 1e9:.2f}B tokens, simulated {p['makespan_h']:.1f}h, ETA {p['eta']}")

    log("=== main")
    jobs = main_jobs(run, state["plan"], a.size, workers, a.n_test, a.n_val, state["plan"]["stage2_problems"],
                     a.n_natural)
    runner.run(jobs, "main")
    rp = os.path.join(run, "report.md")
    log(f"ALL DONE. Report: {rp}" if os.path.exists(rp) else "finished, but report.md is missing -- see logs/report.log")


if __name__ == "__main__":
    main()
