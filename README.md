# Revise and erase only what depends on the error

A language model that writes left to right, keeps its last few lines in a **window**, re-reads that window with
context on both sides, and at every step either **writes** the next token or **fixes** one token inside the
window. After a fix it erases **only the text that depended on the error** and regenerates it.
Tested on iGSM (synthetic grade-school math with a known dependency graph), against six alternatives at equal
training compute.

## Quick start

```bash
pip install -r requirements.txt
bash run_all.sh configs/tiny.yaml     # smoke test: whole pipeline in a few minutes (numbers meaningless)
bash run_all.sh                       # pilot: configs/small.yaml on all visible GPUs (~1.5-2.5 h on 4 GPUs)
bash run_all.sh configs/base.yaml     # GPT-2-small sized model as in the iGSM papers (~10-14 h)
```

`run_all.sh` trains the plain model, trains the revision model at the same FLOPs, evaluates all variants and
writes `runs/<name>/results/results.md` (+ `acc_vs_compute.png`). Every stage is skipped if already finished,
so re-running the command resumes. Overrides go after the config:
`bash run_all.sh configs/small.yaml train.plain_steps=40000 eval.n_test=2000`. Set `NGPU=` to choose the
number of GPUs.

**Storage.** Problems are generated on the fly by CPU workers: nothing is downloaded and no dataset is
stored. Only checkpoints (small: ~50 MB each, base: ~0.35 GB; at most 4 kept) and results are written. Every
checkpoint save checks `disk_limit_gb` (35 GB) for the run directory and stops with a clear error instead of
filling the disk.

**What to watch in the logs** (`runs/<name>/{plain,revise}/log.jsonl`): `eval_greedy_acc` (if the plain
model stays near 0, train longer or use a bigger model), `rollout_err_rate` (how often the model's own
solutions contain a mistake), `err_recall` / `err_fp_rate` (error detection), `fix_acc`, `dep_prec` / `dep_rec`
(dependency head vs. its targets), and in the revision stage `eval_dep_head_acc` vs.
`eval_same_problems_greedy_acc` (the method vs. plain decoding of the same model on 64 problems).

## The seven variants

All revision variants use **one** trained revision model and differ only in what is erased after a fix.

| # | name in code | after fixing token *j* |
|---|---|---|
| 1 | `plain` | no revision (separately trained plain LM, same training FLOPs) |
| 2 | `fix_only` | keep everything after *j* |
| 3 | `erase_suffix` | erase everything after *j*, rewrite |
| 4 | `dep_head` | erase tokens the dependency head predicts depend on *j* **(the method)** |
| 5 | `oracle_dep` | erase the true dependents of *j* from the generator's graph (upper bound) |
| 6 | `dep_head_lines` | as 4, but erase every solution line that contains a dependent token |
| 7 | `random_erase` | erase as many random tokens after *j* as variant 4 would |
| 4b | `confdrop_measure` | as 4, but measure the confidence drop with extra forward passes instead of the head |
| – | `revision_model_no_revision` | diagnostic: the revision model decoding without revising |
| – | `plain_majority@k` | plain model + majority vote over *k* samples, *k* chosen to match the FLOPs of variant 4 |

The report lists accuracy (± standard error), fully-correct-solution rate, inference FLOPs relative to plain
decoding, fixes per problem, erased tokens per fix, and diagnostics computed with the generator's ground truth:
how often a fix hits a real error, and the precision and recall of the erased set against the true
dependents. These are given for the in-distribution test set (op 10–15) and a harder out-of-distribution set
(op 16–20), with accuracy broken down by number of operations.

## How the idea is implemented

**Window and re-reading** (`revise/model.py`). One transformer, two streams packed into one sequence:
the *writer* stream is a normal causal LM over prompt + solution; the *re-read* stream is a copy of the last
`window` solution tokens (64 ≈ 3 lines) that attends to the causal prefix before the window and
bidirectionally to itself. Heads on the re-read stream: `err` (is this token an error?), `fix` (what should be
here? also used to fill erased tokens), and a small bilinear `dep` head: `dep[j, k]` is the probability that
token *k* depends on token *j*.

**Decoding** (`revise/decode.py`). One forward pass per step gives both the next-token distribution and the
window predictions. If the window contains erased tokens, the leftmost one is filled. Otherwise, if some
token has err > `tau_err` (calibrated, see below) and the fix head also prefers another token there, the most likely error is replaced
by the fix head's choice. Otherwise the next token is written. Never more than one
action per step. Because `dep[j, :]` comes out of the same pass that chose *j*, the method needs no extra pass
to find dependents. Erasure: a dependent run that reaches the end of the text is truncated and rewritten.
Runs in the middle become `<mask>` tokens and are filled in place (long runs get `slack` extra masks that may be
filled with `<null>`, so regenerated text can be shorter). A budget (`max_fixes`) caps revisions per problem.

**Finding dependents by confidence drop** (`confdrop_targets` in `revise/train.py`, `_measure_confdrop` in
`decode.py`). Confidence is the writer's log-probability of each later token given the text before it, on the
model's own text vs. the same text with the error fixed; a drop above `confdrop_delta` (2 nats) marks the
token as dependent. A single comparison misses indirect dependents: in `q = p + 2 = 7 + 2 = 9`, after fixing p's
value the `7` loses confidence but the `9` is still consistent with `7 + 2`. So the dependents found are replaced
by what the model would now write, and the measurement is repeated (up to `confdrop_iters` passes) until nothing
new drops. These measured labels are the training target of the dependency head (`revision.dep_target:
confdrop`); `dep_target: oracle` trains it on the generator's graph instead. (`confdrop_mode: reread` uses the
re-read fix head as the confidence instead. That head sees the token it scores, so it learns "the visible token
is fine" and barely reacts to an earlier fix: in a test run it produced no dependents at all.)
`tests/test_confdrop.py` checks the measurement with a scripted, perfectly consistent writer. It must find
every token that changes and never flag a token outside the true dependency graph.

**Error threshold.** Errors are rare among window tokens, so the error head's probabilities are low and a fixed
0.5 threshold may never fire. With `eval.policy.tau_err: auto` the threshold is calibrated on the model's own
greedy solutions to 256 held-out problems (different seed from the test sets). It is set so that about
`calib_false_per_problem` (0.5) false fixes per solution are expected on text that has no error yet. The same
threshold is used for every variant; the detection recall at that threshold is printed in the report.

**Training** (`revise/train.py`).
1. *Plain stage:* causal LM on correct solutions. Its FLOPs define the budget; the checkpoint at 50 % is saved.
2. *Revision stage:* starts from that checkpoint and spends the remaining 50 % of the FLOPs. Rollouts and
   confidence-drop passes are counted too, so the total compute equals the plain model's. Each batch mixes:
   - **correct solutions**: LM loss, plus re-read windows with no errors;
   - **masked gaps**: windows of correct solutions in which dependency-shaped, span, line or random token sets
     are masked;
   - **the model's own errors**: every 50 steps each GPU generates 256 solutions (greedy and sampled), the
     checker finds the first mistake, and windows containing it (detected right away or up to a window later)
     train the error head, the fix head (target = set of valid tokens) and the dependency row of the mistake.
   - The learning-rate schedule is a function of the fraction of the compute budget used, so both models follow
     the same schedule.

**iGSM** (`revise/igsm/`). A re-implementation of the iGSM generator (Ye et al., *Physics of Language Models
2.1*, 2024) rather than the official code: a 4-layer structure graph of items, instance parameters (edges),
abstract parameters (category totals as sums of products), and a random dependency graph. Arithmetic is
mod 23, problem sentences are shuffled and include distractors, and solutions compute only the necessary
parameters in a random topological order with temporary variables, e.g.
`Define Film Studio 's School Daypack as g ; R = W + B = 13 + 7 = 20 ; so g = R + 12 = 20 + 12 = 9 .`
A closed word-level vocabulary (362 tokens) is used. `checker.py` labels any (partial, model-written) solution
token by token, with *consistency* semantics: a value correctly computed from an earlier wrong value is not a
new error, it depends on it. It also gives the set of valid tokens at each error and the token-level
information-flow graph: which value was computed from which operands, which formula belongs to which
parameter, and which letter refers to which definition. That graph drives the oracle (variant 5) and the
erase precision/recall diagnostics.

## Compute accounting and caveats

- **Training compute** is matched exactly in FLOPs (6·N per trained token, 2·N per inference token for
  rollouts and confidence-drop passes).
- **Inference compute** is reported per problem as if the unchanged prefix were KV-cached. Re-reading a
  64-token window at every step costs roughly 30× a plain greedy decode for these problem sizes. This is why
  `plain_majority@k` at matched FLOPs is included, and why `eval.policy.check_every` (re-read every k steps) and
  `eval.sweep_max_fixes` (accuracy-vs-compute curves) exist.
- The current decoder recomputes the full forward pass at every step (simple and exact), so wall-clock time is
  higher than the FLOP numbers suggest.
- Computing an unnecessary parameter counts as an error (`data.strict_necessary: true`).
- Length-changing regeneration in the middle of the text is limited by `revision.slack`.

## Layout

```
run_all.sh                 one command: train plain -> train revision -> evaluate -> report
configs/{tiny,small,base}.yaml
revise/igsm/generator.py   iGSM-style generator (dependency graph, values, problem/solution text)
revise/igsm/checker.py     labeller: root errors, valid fixes, token dependency DAG, final answer
revise/model.py            two-stream transformer, attention mask, heads, KV cache
revise/examples.py         training examples (clean / masked / own-error windows), erasure, collate
revise/data.py             on-the-fly problem generation workers
revise/rollout.py          KV-cached generation, rollout buffer of the model's own mistakes
revise/train.py            DDP training, FLOP budget, losses, confidence-drop labels
revise/decode.py           revision decoding and all variants
revise/evaluate.py         distributed evaluation, summaries
revise/report.py           results.md and plot
tests/                     python -m pytest -q tests
```
