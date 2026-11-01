"""Phase 0 with one command: data -> 4 GPU lanes in parallel -> report.

  lane base_test : base model, test set (neutral x8, 4 strategies x2, all verifications)
  lane noise     : noise scale on val, then noisy members solve and verify (one noisy checkpoint
                   on disk at a time, deleted right after use)
  lanes pool_i   : base model, shard i of the train pool (4 strategies x2)
Re-running the same command skips everything whose output already exists.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time

from . import data as D
from . import report

PY = sys.executable


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def dir_gb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 1e9


class Lanes:
    def __init__(self, run, gpus, disk_limit_gb):
        self.run, self.gpus, self.disk_limit = run, gpus, disk_limit_gb
        self.locks = {g: threading.Lock() for g in gpus}
        self.failed = []
        os.makedirs(os.path.join(run, "logs"), exist_ok=True)

    def sh(self, name, cmd, gpu, output):
        """Run one step unless `output` exists. Returns True on success."""
        if output and os.path.exists(output):
            return True
        if dir_gb(self.run) > self.disk_limit:
            log(f"[{name}] disk limit reached ({dir_gb(self.run):.1f} GB) -- skipping")
            self.failed.append(name)
            return False
        env = dict(os.environ, PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu),
                   TOKENIZERS_PARALLELISM="false")
        env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")      # its JIT build needs nvcc, absent on many servers
        lf = open(os.path.join(self.run, "logs", f"{name}.log"), "a")
        lf.write(f"\n### {time.strftime('%F %T')} {' '.join(cmd)}\n")
        lf.flush()
        lock = self.locks.get(gpu)
        if lock:
            lock.acquire()
        try:
            log(f"[{name}] start" + ("" if gpu is None else f" on gpu {gpu}"))
            t0 = time.time()
            rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        finally:
            if lock:
                lock.release()
            lf.close()
        ok = rc == 0 and (not output or os.path.exists(output))
        log(f"[{name}] {'done' if ok else 'FAILED (see logs/' + name + '.log)'} in {(time.time() - t0) / 60:.1f} min")
        if not ok:
            self.failed.append(name)
        return ok


def gen_cmd(job, model, a, out, extra=(), eager=None):
    eager = a.eager if eager is None else eager
    return [PY, "-m", "phase0.gen", job, "--model", model, "--data", os.path.join(a.run, "data"), "--out", out,
            "--backend", a.backend, "--eager", str(int(eager)), "--max_tokens", str(a.max_tokens), *extra]


def lane_noise(L, a, model, gpu, state):
    nd = os.path.join(a.run, "noise")
    os.makedirs(nd, exist_ok=True)
    tmp = os.path.join(a.run, "tmp_noisy_model")
    # noisy members: small generation volume, so skip CUDA-graph capture on every model load

    def noisy(name, sigma, seed, cmd, output):
        if os.path.exists(output):
            return True
        free = shutil.disk_usage(a.run).free / 1e9
        if free < a.min_free_gb:
            log(f"[noise] only {free:.1f} GB free, need {a.min_free_gb} GB for a noisy checkpoint -- stopping noise lane")
            L.failed.append(name)
            return False
        ok = L.sh(f"{name}_make", [PY, "-m", "phase0.backend", "make_noisy", "--base", model, "--out", tmp,
                                   "--sigma", str(sigma), "--seed", str(seed)], None, None)
        ok = ok and L.sh(name, cmd, gpu, output)
        shutil.rmtree(tmp, ignore_errors=True)
        return ok

    base_val = os.path.join(nd, "val_base.json")
    if not L.sh("noise_val_base", gen_cmd("noise_val", model, a, nd, ["--tag", "base"], eager=1), gpu, base_val):
        return
    accs = {}
    for s in a.sigmas:
        f = os.path.join(nd, f"val_s{s}.json")
        if not noisy(f"noise_val_s{s}", s, 1000, gen_cmd("noise_val", tmp, a, nd, ["--tag", f"s{s}"], eager=1), f):
            return
        accs[s] = json.load(open(f))["acc"]
    if not accs:
        return
    base = json.load(open(base_val))["acc"]
    ok = [s for s in accs if accs[s] >= base - 0.03]
    sigma = max(ok) if ok else min(accs)
    state["noise"] = {"base_val_acc": base, "val_acc": accs, "sigma": sigma,
                      "note": "" if ok else "все σ ухудшают точность больше чем на 3 п.; взят минимальный"}
    json.dump(state["noise"], open(os.path.join(nd, "choice.json"), "w"))
    log(f"[noise] base val acc {base:.3f}, by sigma {accs} -> sigma {sigma}")
    for k in range(a.members):
        if not noisy(f"noise_solve_{k}", sigma, k, gen_cmd("noise_solve", tmp, a, nd, ["--tag", str(k), "--seed", str(1000 * (k + 1))], eager=1),
                     os.path.join(nd, f"solve_{k}.jsonl.gz")):
            return                        # stop the lane at the first failure instead of retrying every member
    for j in range(a.members):
        if not noisy(f"noise_verify_{j}", sigma, j,
              gen_cmd("noise_verify", tmp, a, nd, ["--tag", str(j), "--noise_dir", nd, "--members", str(a.members)],
                      eager=1), os.path.join(nd, f"verify_{j}.json")):
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/phase0")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--backend", default="auto", choices=["auto", "vllm", "hf"])
    ap.add_argument("--eager", type=int, default=0)
    ap.add_argument("--n_test", type=int, default=1500)
    ap.add_argument("--n_val", type=int, default=300)
    ap.add_argument("--n_pool", type=int, default=12000)
    ap.add_argument("--pool_shards", type=int, default=2)
    ap.add_argument("--sources", nargs="+", default=["gsm", "math", "synth"])
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.002, 0.005, 0.01, 0.02])
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=1200, help="max tokens per solution")
    ap.add_argument("--disk_limit_gb", type=float, default=30.0, help="max size of the run directory")
    ap.add_argument("--min_free_gb", type=float, default=6.0, help="free space needed for a noisy checkpoint")
    ap.add_argument("--smoke", action="store_true", help="tiny sizes, checks every step end to end")
    a = ap.parse_args()
    if a.smoke:
        a.n_test, a.n_val, a.n_pool, a.sigmas, a.members, a.eager = 24, 6, 16, [0.01], 2, 1
        a.max_tokens = min(a.max_tokens, 512)
    a.run = os.path.abspath(a.run)
    os.makedirs(os.path.join(a.run, "data"), exist_ok=True)
    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    t0 = time.time()
    sp = os.path.join(a.run, "state.json")
    state = json.load(open(sp)) if os.path.exists(sp) else {}

    # ---- data and model
    dd = os.path.join(a.run, "data")
    if not os.path.exists(os.path.join(dd, "pool.jsonl")):
        log("building datasets (downloads GSM8K / GSM-Symbolic / MATH on first run)")
        sets = D.build(a.n_test, a.n_val, a.n_pool, sources=tuple(a.sources))
        for k in ("test", "val", "pool"):
            D.save_jsonl(os.path.join(dd, f"{k}.jsonl"), sets[k])
        state["data_notes"] = sets["notes"]
        state["sizes"] = {k: len(sets[k]) for k in ("test", "val", "pool")}
        log(f"data: {state['sizes']}; notes: {sets['notes']}")
    model = a.model
    if not os.path.isdir(model):
        from huggingface_hub import snapshot_download
        log(f"downloading {model} (once, to the Hugging Face cache)")
        model = snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model",
                                                         "tokenizer*", "merges*", "vocab*"])
    state.update(model=a.model, model_path=model, backend=a.backend, smoke=a.smoke, members=a.members)
    json.dump(state, open(sp, "w"), indent=1)

    # ---- GPU lanes
    L = Lanes(a.run, gpus, a.disk_limit_gb)
    lanes = [("base_test", lambda g: L.sh("base_test", gen_cmd("base_test", model, a, os.path.join(a.run, "test")), g,
                                          os.path.join(a.run, "test", "verify_strategies.json"))),
             ("noise", lambda g: lane_noise(L, a, model, g, state))]
    for i in range(a.pool_shards):
        lanes.append((f"pool_{i}", lambda g, i=i: L.sh(
            f"pool_{i}", gen_cmd("pool", model, a, os.path.join(a.run, "pool"),
                                 ["--shard", str(i), "--nshards", str(a.pool_shards)]), g,
            os.path.join(a.run, "pool", f"pool_{i}.jsonl.gz"))))
    threads = []
    for n, (name, fn) in enumerate(lanes):
        t = threading.Thread(target=fn, args=(gpus[n % len(gpus)],), name=name)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    json.dump(state, open(sp, "w"), indent=1)

    # ---- report
    state["failed"] = L.failed
    state["hours"] = (time.time() - t0) / 3600
    json.dump(state, open(sp, "w"), indent=1)
    path = report.build(a.run)
    log(f"report: {path}" + (f"  (failed steps: {L.failed})" if L.failed else ""))
    log(f"disk used by run dir: {dir_gb(a.run):.2f} GB")
    if L.failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
