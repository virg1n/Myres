"""Extra baseline: noise in the base weights with larger sigma (phase 0 stopped at 0.02 without any
accuracy loss). Reuses the phase-0 noise lane on the phase-0 test / val sets."""
import argparse
import os
import shutil
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase0.run import Lanes, lane_noise  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase0", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=1200)
    a = ap.parse_args()
    run = os.path.abspath(a.run)
    os.makedirs(os.path.join(run, "data"), exist_ok=True)
    for f in ("test.jsonl", "val.jsonl"):
        shutil.copy(os.path.join(a.phase0, "data", f), os.path.join(run, "data", f))
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    args = types.SimpleNamespace(run=run, sigmas=a.sigmas, members=a.members, min_free_gb=6.0, backend="auto",
                                 eager=0, max_tokens=a.max_tokens)
    L = Lanes(run, [gpu], disk_limit_gb=1e9)
    state = {}
    lane_noise(L, args, a.model, gpu, state)
    ok = all(os.path.exists(os.path.join(run, "noise", f"verify_{j}.json")) for j in range(a.members))
    if not ok:
        raise SystemExit(f"noise lane incomplete: failed {L.failed}")
    open(os.path.join(run, "DONE"), "w").close()


if __name__ == "__main__":
    main()
