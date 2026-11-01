"""Phase-0 report on fabricated outputs with known structure (CPU, no model needed).

    python3 -m tests.test_phase0_report

temperature : errors share one latent "difficulty" and mostly the same wrong answer
strategies  : each strategy has its own blind spots -> decorrelated errors, different wrong answers;
              strategy verifiers see other strategies' errors well, their own poorly
Expected: decision 1 flags that prompts decorrelate errors, verifier AUROC is high, the
'off-diagonal' TPR beats the diagonal for strategies."""
import gzip
import json
import os
import random
import tempfile

import numpy as np

from phase0 import report


def write_gz(path, rows):
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    rng = np.random.default_rng(0)
    run = tempfile.mkdtemp()
    for d in ("data", "test", "noise", "pool"):
        os.makedirs(os.path.join(run, d))
    N = 600
    test = [{"id": f"t{n}", "source": "synth", "bin": f"synth_{4 + n % 5}", "question": "q", "answer": "7"}
            for n in range(N)]
    with open(os.path.join(run, "data", "test.jsonl"), "w") as f:
        for r in test:
            f.write(json.dumps(r) + "\n")
    hard = rng.random(N)

    def sol(ok, wrong_id):
        return ("\\boxed{7}" if ok else f"\\boxed{{{wrong_id}}}"), "7" if ok else str(wrong_id)

    # temperature: correctness driven by the shared latent, wrong answer usually the same (100)
    neu = []
    for n in range(N):
        texts, ans, cor = [], [], []
        for s in range(8):
            ok = bool(rng.random() > 0.15 + 0.6 * hard[n])
            t, a = sol(ok, 100 if rng.random() < 0.8 else 100 + s)
            texts.append(t), ans.append(a), cor.append(ok)
        neu.append({"id": f"t{n}", "mode": "neutral", "texts": texts, "answers": ans, "correct": cor})
    write_gz(os.path.join(run, "test", "test_neutral.jsonl.gz"), neu)
    # strategies: own blind spot per strategy
    blind = rng.random((4, N)) < 0.35
    st_cor = np.zeros((4, 2, N), bool)
    for k in range(4):
        rows = []
        for n in range(N):
            texts, ans, cor = [], [], []
            for s in range(2):
                ok = bool(not blind[k, n] and rng.random() > 0.1)
                t, a = sol(ok, 200 + k)
                texts.append(t), ans.append(a), cor.append(ok)
                st_cor[k, s, n] = ok
            rows.append({"id": f"t{n}", "mode": k, "texts": texts, "answers": ans, "correct": cor})
        write_gz(os.path.join(run, "test", f"test_strategy{k}.jsonl.gz"), rows)
    tc = np.array([[r["correct"][i] for r in neu] for i in range(4)])
    pn = [list(map(float, np.clip(np.where(tc[i], 0.3, 0.7) + rng.normal(0, 0.15, N), 0, 1))) for i in range(4)]
    json.dump({"p_no": pn, "missing": 0}, open(os.path.join(run, "test", "verify_neutral.json"), "w"))
    mat = [[None] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            strength = 0.1 if i == j else 0.35
            err = ~st_cor[i, 0]
            mat[i][j] = list(map(float, np.clip(0.5 + np.where(err, strength, -strength) + rng.normal(0, 0.15, N), 0, 1)))
    json.dump({"p_no": mat, "missing": 0}, open(os.path.join(run, "test", "verify_strategies.json"), "w"))
    json.dump({"model": "fake", "backend": "none"}, open(os.path.join(run, "state.json"), "w"))
    path = report.build(run)
    text = open(path).read()
    print("\n".join(text.splitlines()[:30]))
    assert "⚠️ промпты заметно разводят ошибки" in text, "decision 1 should flag decorrelation by prompts"
    assert "✅ работает" in text, "verifier AUROC should pass"
    s = json.load(open(os.path.join(run, "summary_phase0.json")))["summary"]
    assert s["strategies"]["ver_tpr_off"] > s["strategies"]["ver_tpr_diag"] + 0.2
    print("phase-0 report on fabricated data: ok")


if __name__ == "__main__":
    random.seed(0)
    main()
