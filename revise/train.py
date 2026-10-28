"""Training.

  stage plain  : standard causal LM on correct solutions (variant 1). Its total
                 FLOPs define the compute budget; a checkpoint at `branch_frac`
                 of training is saved as the starting point of the revision run.
  stage revise : starts from the plain model at `branch_frac` and spends the
                 *remaining* FLOPs (same total as the plain run, rollout and
                 confidence-drop passes included) on the mixed objective:
                 correct solutions + the model's own erroneous rollouts +
                 solutions with masked gaps.

Run with torchrun, e.g.
  torchrun --nproc_per_node=4 -m revise.train --config configs/small.yaml --stage plain
"""

import argparse
import json
import os
import random
import time

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from .decode import calibrate_tau_err, decode_plain, decode_revision, make_policy
from .examples import collate, error_example
from .data import make_loader
from .igsm.generator import GenConfig, ProblemSampler, make_test_set
from .igsm.tokenizer import Tokenizer
from .model import ModelConfig, RevisionLM, build_mask
from .rollout import RolloutBuffer, collect_rollouts
from .utils import (Logger, all_reduce_sum, amp_dtype, barrier, cleanup_dist, init_dist, is_dist, load_config,
                    lr_at, save_checkpoint, unwrap)


def build_model(cfg, tok):
    return RevisionLM(ModelConfig.from_dict(cfg["model"], len(tok)))


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def forward_batch(model, batch, amp):
    mask = build_mask(batch["pos"], batch["is_copy"], batch["wstart"], batch["wid"], batch["valid"])
    with torch.autocast(batch["ids"].device.type, dtype=amp, enabled=amp is not None):
        h = model(batch["ids"], batch["pos"], batch["is_copy"].long(), mask)
    return h


def set_ce(logits, sets):
    """-log sum_{v in set} p(v); sets padded with -1."""
    lse = torch.logsumexp(logits, -1)
    g = logits.gather(-1, sets.clamp(min=0)).masked_fill(sets < 0, float("-inf"))
    return lse - torch.logsumexp(g, -1)


def compute_losses(m, h, batch, cfg, step, stage):
    """m: unwrapped model. Returns total loss, metrics, and per-copy-position logp of the written token."""
    t = cfg["train"]
    lw = t["loss_w"]
    d = h.shape[-1]
    hf = h.reshape(-1, d)
    metrics = {}
    lm_t = batch["lm_tgt"].reshape(-1)
    sel = lm_t >= 0
    lm_logits = m.lm_logits(hf[sel]).float()
    loss_lm = F.cross_entropy(lm_logits, lm_t[sel]) if sel.any() else h.sum() * 0
    metrics["loss_lm"] = loss_lm.item()
    total = lw["lm"] * loss_lm
    if stage == "plain":
        return total, metrics, None

    # error head
    et = batch["err_tgt"].reshape(-1)
    esel = et >= 0
    if esel.any():
        el = m.err_logits(hf[esel]).float()
        pw = torch.tensor(t["err_pos_weight"], device=h.device)
        loss_err = F.binary_cross_entropy_with_logits(el, et[esel].float(), pos_weight=pw)
        with torch.no_grad():
            p = torch.sigmoid(el)
            pos = et[esel] == 1
            if pos.any():
                metrics["err_p_pos"] = p[pos].mean().item()
                metrics["err_recall"] = (p[pos] > 0.5).float().mean().item()
            metrics["err_p_neg"] = p[~pos].mean().item() if (~pos).any() else 0.0
            metrics["err_fp_rate"] = (p[~pos] > 0.5).float().mean().item() if (~pos).any() else 0.0
    else:
        loss_err = h.sum() * 0
    metrics["loss_err"] = loss_err.item()

    # fix / fill head
    fp = batch["fix_pos"]
    if fp.numel():
        fl = m.fix_logits(hf[fp]).float()
        ce = set_ce(fl, batch["fix_sets"])
        w = batch["fix_w"]
        loss_fix = (ce * w).sum() / w.sum().clamp(min=1e-6)
        with torch.no_grad():
            pred = fl.argmax(-1, keepdim=True)
            hit = ((batch["fix_sets"] == pred) & (batch["fix_sets"] >= 0)).any(-1)
            strong = w >= 0.999
            if strong.any():
                metrics["fix_acc"] = hit[strong].float().mean().item()
    else:
        loss_fix = h.sum() * 0
    metrics["loss_fix"] = loss_fix.item()

    # dependency head
    loss_dep = h.sum() * 0
    if batch["dep_q"].numel():
        hq = hf[batch["dep_q"]]
        hk = hf[batch["dep_k"].clamp(min=0)]
        dl = m.dep_logits(hq, hk).float()
        dt = batch["dep_t"]
        dsel = dt >= 0
        if dsel.any():
            loss_dep = F.binary_cross_entropy_with_logits(dl[dsel], dt[dsel].float())
            with torch.no_grad():
                pr = dl[dsel] > 0
                tg = dt[dsel] == 1
                metrics["dep_prec"] = ((pr & tg).sum() / pr.sum().clamp(min=1)).item()
                metrics["dep_rec"] = ((pr & tg).sum() / tg.sum().clamp(min=1)).item()
                metrics["dep_pos_frac"] = tg.float().mean().item()
    metrics["loss_dep"] = loss_dep.item()
    total = total + lw["err"] * loss_err + lw["fix"] * loss_fix + lw["dep"] * loss_dep
    # keep every revision parameter in the graph (DDP)
    total = total + 0.0 * sum(p.sum() for p in m.revision_parameters())
    return total, metrics, None


@torch.no_grad()
def confdrop_targets(m, h, batch, cfg, amp, mask_id):
    """Fill batch['dep_t'] rows marked for confidence-drop measurement:
    tokens after the error whose confidence drops by more than `delta` nats
    once the error is fixed are labelled dependent. Returns tokens processed."""
    if not batch["confdrop"]:
        return 0
    if cfg["revision"].get("confdrop_mode", "writer") == "writer":
        return _confdrop_writer(m, h, batch, cfg, amp)
    return _confdrop_reread(m, h, batch, cfg, amp, mask_id)


@torch.no_grad()
def _confdrop_writer(m, h, batch, cfg, amp):
    """Confidence = the writer's log p(token | text before it).
    before: on the model's own text (from the training forward pass, free);
    after:  with the error replaced by its fix. Dependents found in a pass are
    replaced by what the model would now write ("virtual repair") and the
    measurement is repeated, so that indirect dependents -- e.g. the 9 in
    '= 7 + 2 = 9' after the 7 itself turned out to depend on the fix -- are
    found as well."""
    rows = batch["confdrop"]
    r_cfg = cfg["revision"]
    delta, iters = r_cfg["confdrop_delta"], r_cfg["confdrop_iters"]
    dev = batch["ids"].device
    row_i, rb, off, n, j, ft = (torch.tensor(c, device=dev) for c in zip(*rows))
    R, Wm = len(rows), int(n.max())
    sub = {k: batch[k][rb] for k in ("ids", "pos", "is_copy", "wstart", "wid", "valid")}
    wst = batch["wstart"][rb, off]  # causal position of each window's first token
    kk = torch.arange(Wm, device=dev)
    in_win = kk[None] < n[:, None]
    cpos = torch.where(in_win, wst[:, None] + kk[None], torch.ones_like(kk)[None])
    later = in_win & (kk[None] > j[:, None])
    rr = torch.arange(R, device=dev)[:, None].expand(R, Wm)
    ar = torch.arange(R, device=dev)

    def logp_next(hh, ids, sel):
        logits = m.lm_logits(hh[sel[:, None], cpos - 1]).float()
        lp = torch.log_softmax(logits, -1).gather(-1, ids[sel[:, None], cpos].unsqueeze(-1)).squeeze(-1)
        return lp, logits.argmax(-1)

    before, _ = logp_next(h.detach(), batch["ids"], rb)
    ids = sub["ids"].clone()
    ids[ar, wst + j] = ft
    dep = torch.zeros(R, Wm, dtype=torch.bool, device=dev)
    active = torch.ones(R, dtype=torch.bool, device=dev)
    processed = 0
    for it in range(iters):
        act = active.nonzero().squeeze(1)  # only rows that still found new dependents
        part = {k: v[act] for k, v in sub.items()}
        part["ids"] = ids[act]
        processed += int(part["valid"].sum())
        hh = forward_batch(m, part, amp)
        after = torch.full_like(before, float("inf"))
        pred = torch.zeros_like(ids[:, :Wm])
        logits = m.lm_logits(hh[torch.arange(len(act), device=dev)[:, None], cpos[act] - 1]).float()
        after[act] = torch.log_softmax(logits, -1).gather(-1, ids[act[:, None], cpos[act]].unsqueeze(-1)).squeeze(-1)
        pred[act] = logits.argmax(-1)
        new = later & ~dep & ((before - after) > delta) & active[:, None]
        if not bool(new.any()):
            break
        dep |= new
        active = new.any(1)
        ids[rr[new], cpos[new]] = pred[new]
    tgt = torch.where(later, dep.long(), torch.full_like(dep, -1, dtype=torch.long))
    batch["dep_t"][row_i, :Wm] = tgt
    return processed


@torch.no_grad()
def _confdrop_reread(m, h, batch, cfg, amp, mask_id):
    """Alternative: confidence = re-read fix head's log p(written token);
    dependents found so far are masked between passes."""
    rows = batch["confdrop"]
    r_cfg = cfg["revision"]
    delta, iters = r_cfg["confdrop_delta"], r_cfg["confdrop_iters"]
    dev = batch["ids"].device
    row_i, rb, off, n, j, ft = (torch.tensor(c, device=dev) for c in zip(*rows))
    R, Wm = len(rows), int(n.max())
    sub = {k: batch[k][rb] for k in ("ids", "pos", "is_copy", "wstart", "wid", "valid")}
    kk = torch.arange(Wm, device=dev)
    in_win = kk[None] < n[:, None]
    pidx = torch.where(in_win, off[:, None] + kk[None], torch.zeros_like(kk)[None])
    later = in_win & (kk[None] > j[:, None])
    rr = torch.arange(R, device=dev)[:, None].expand(R, Wm)

    def logp_written(hh, ids, sel):
        x = hh[sel[:, None], pidx]
        lp = torch.log_softmax(m.fix_logits(x).float(), -1)
        return lp.gather(-1, ids[sel[:, None], pidx].unsqueeze(-1)).squeeze(-1)

    # "before" comes from the training forward pass (same weights, no extra cost)
    before = logp_written(h.detach(), batch["ids"], rb)
    ids = sub["ids"].clone()
    ids[torch.arange(R, device=dev), off + j] = ft
    dep = torch.zeros(R, Wm, dtype=torch.bool, device=dev)
    active = torch.ones(R, dtype=torch.bool, device=dev)
    processed = 0
    for it in range(iters):
        processed += int(sub["valid"].sum())
        after = logp_written(forward_batch(m, {**sub, "ids": ids}, amp), ids, torch.arange(R, device=dev))
        new = later & ~dep & ((before - after) > delta) & active[:, None]
        if not bool(new.any()):
            break
        dep |= new
        active = new.any(1)
        ids[rr[new], pidx[new]] = mask_id
    tgt = torch.where(later, dep.long(), torch.full_like(dep, -1, dtype=torch.long))
    batch["dep_t"][row_i, :Wm] = tgt
    return processed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", choices=["plain", "revise"], required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    rank, world, device = init_dist()
    t = cfg["train"]
    seed = cfg["seed"] + (0 if args.stage == "plain" else 7919) + 104729 * rank
    torch.manual_seed(seed)
    random.seed(seed)
    amp = amp_dtype(cfg) if device.type == "cuda" else None

    stage_dir = os.path.join(cfg["out_dir"], args.stage)
    plain_dir = os.path.join(cfg["out_dir"], "plain")
    if rank == 0:
        os.makedirs(stage_dir, exist_ok=True)
        with open(os.path.join(cfg["out_dir"], "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
    if os.path.exists(os.path.join(stage_dir, "final.pt")):
        if rank == 0:
            print(f"[{args.stage}] final.pt exists, skipping training")
        cleanup_dist()
        return
    log = Logger(os.path.join(stage_dir, "log.jsonl"), rank)
    tok = Tokenizer()
    model = build_model(cfg, tok).to(device)
    N = model.n_params()
    if rank == 0:
        print(f"model: {N / 1e6:.2f}M non-embedding params, vocab {len(tok)}, world {world}, device {device}")

    # ---- budget / init --------------------------------------------------
    step, flops = 0, 0.0
    if args.stage == "plain":
        for p in model.revision_parameters():
            p.requires_grad_(False)
        total_steps = t["plain_steps"]
        branch_step = max(1, int(round(t["branch_frac"] * total_steps)))
        budget = None
    else:
        with open(os.path.join(plain_dir, "budget.json")) as f:
            budget = json.load(f)
        ck = torch.load(os.path.join(plain_dir, "branch.pt"), map_location="cpu")
        model.load_state_dict(ck["model"])
        flops = budget["branch_flops"]
        if rank == 0:
            print(f"revise: init from plain step {ck['step']}, flops {flops:.3e} / budget {budget['total_flops']:.3e}")

    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": t["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=t["lr"], betas=tuple(t["betas"]), fused=device.type == "cuda")

    last_path = os.path.join(stage_dir, "last.pt")
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, flops = ck["step"], ck["flops"]
        if rank == 0:
            print(f"resumed {args.stage} from step {step}")

    raw = model
    if t.get("compile"):
        model = torch.compile(model, dynamic=True)
    ddp = DDP(model, device_ids=[device.index] if device.type == "cuda" else None) if is_dist() else model

    # ---- data -----------------------------------------------------------
    B = t["batch_size"]
    n_err = 0 if args.stage == "plain" else int(round(B * t["self_error_frac"]))
    loader = iter(make_loader(cfg, args.stage, B - n_err, seed, t["num_workers"]))
    buffer = RolloutBuffer(t["buffer_size"], seed)
    gcfg = GenConfig.from_dict(cfg["data"]["gen"])
    sampler = ProblemSampler(gcfg, cfg["data"]["op_rollout"], seed + 1, tok)
    rrng = random.Random(seed + 2)
    eval_set = make_test_set(gcfg, cfg["data"]["op_test_id"], t["eval_n"], cfg["seed"] + 99, tok) if rank == 0 else None

    def do_rollouts(n):
        entries, stats, proc = collect_rollouts(raw, sampler, tok, n, device, t["rollout_temps"],
                                                t["rollout_max_new"], amp, t["rollout_batch"],
                                                cfg["data"]["op_rollout"], rrng, cfg["data"]["strict_necessary"])
        for e in entries:
            buffer.add(e)
        raw.train()
        return stats, proc

    def done():
        if args.stage == "plain":
            return step >= total_steps
        return flops >= budget["total_flops"]

    t_start = time.time()
    tok_count, last_log = 0, time.time()
    roll_stats = {}
    if args.stage == "revise":
        tries = 0
        while all_reduce_sum(len(buffer) < t["min_buffer"], device) > 0 and tries < 20:
            roll_stats, proc = do_rollouts(t["rollout_problems"])
            flops += 2.0 * N * all_reduce_sum(proc, device)
            tries += 1
        if rank == 0:
            print(f"initial buffer: {len(buffer)} erroneous rollouts ({roll_stats})")

    raw.train()
    while not done():
        if (time.time() - t_start) / 3600 > t["max_hours"]:
            break
        # rollouts of the current model
        if args.stage == "revise" and step > 0 and step % t["rollout_every"] == 0:
            roll_stats, proc = do_rollouts(t["rollout_problems"])
            flops += 2.0 * N * all_reduce_sum(proc, device)

        exs = next(loader)
        if n_err:
            errs = buffer.sample(n_err)
            exs = exs + [error_example(e, rrng, tok, cfg["revision"]) for e in errs]
        batch = to_device(collate(exs, tok.pad), device)

        frac = step / total_steps if args.stage == "plain" else flops / budget["total_flops"]
        lr = lr_at(frac, cfg)
        if args.stage == "revise":
            lr *= min(1.0, (step + 1) / max(1, t["rewarmup_steps"]))
        for g in opt.param_groups:
            g["lr"] = lr

        h = forward_batch(ddp, batch, amp)
        extra = 0
        if args.stage == "revise" and batch["confdrop"] and step >= t["dep_warmup_steps"]:
            extra = confdrop_targets(raw, h, batch, cfg, amp, tok.mask)
        loss, metrics, _ = compute_losses(raw, h, batch, cfg, step, args.stage)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, t["grad_clip"])
        opt.step()
        step += 1

        ntok = int(batch["valid"].sum())
        tok_count += ntok
        flops += all_reduce_sum(6.0 * N * ntok + 2.0 * N * extra, device)

        if args.stage == "plain" and step == branch_step and rank == 0:
            save_checkpoint(os.path.join(stage_dir, "branch.pt"),
                            {"model": raw.state_dict(), "step": step, "flops": flops}, cfg)
            with open(os.path.join(stage_dir, "branch.json"), "w") as f:
                json.dump({"branch_step": step, "branch_flops": flops}, f)

        if step % t["log_every"] == 0:
            dt = time.time() - last_log
            last_log = time.time()
            rec = {"stage": args.stage, "step": step, "lr": lr, "loss": loss.item(), "gnorm": gn.item(),
                   "flops": flops, "frac": round(frac, 4), "tok_per_s": tok_count * world / max(dt, 1e-6),
                   **metrics}
            if args.stage == "revise":
                rec.update({"buffer": len(buffer), **roll_stats})
            tok_count = 0
            log.log(rec)

        if step % t["eval_every"] == 0 and rank == 0:
            res = decode_plain(raw, eval_set, tok, device, batch_size=t["rollout_batch"], amp_dtype=amp,
                               max_sol_len=t["rollout_max_new"])
            rec = {"stage": args.stage, "step": step, "eval_greedy_acc": sum(r["correct"] for r in res) / len(res)}
            if args.stage == "revise" and t.get("eval_revision_n", 0):
                nr = min(t["eval_revision_n"], len(eval_set) // 2)
                cal = calibrate_tau_err(raw, eval_set[-nr:], tok, device, cfg["revision"]["window"], amp,
                                        cfg["eval"].get("calib_false_per_problem", 0.5), batch_size=64,
                                        max_sol_len=t["rollout_max_new"])
                pol = make_policy("dep_head", {**cfg["eval"]["policy"], "window": cfg["revision"]["window"],
                                               "slack": cfg["revision"]["slack"], "diagnostics": True,
                                               "tau_err": cal["tau_err"]})
                rr = decode_revision(raw, eval_set[:nr], pol, tok, device,
                                     batch_size=cfg["eval"]["batch_size"], amp_dtype=amp)
                rec.update({"eval_tau_err": cal["tau_err"], "eval_detect_recall": cal["detect_recall"]})
                ev = [e for r in rr for e in r["events"]]
                rec.update({"eval_dep_head_acc": sum(r["correct"] for r in rr) / len(rr),
                            "eval_same_problems_greedy_acc": sum(r["correct"] for r in res[: len(rr)]) / len(rr),
                            "eval_fixes_per_problem": len(ev) / len(rr),
                            "eval_fix_precision": sum(e["true_err"] for e in ev) / max(1, len(ev))})
            log.log(rec)
            raw.train()

        if step % t["save_every"] == 0 and rank == 0:
            save_checkpoint(last_path, {"model": raw.state_dict(), "opt": opt.state_dict(), "step": step,
                                        "flops": flops}, cfg)
        barrier() if step % t["save_every"] == 0 else None

    if rank == 0:
        save_checkpoint(os.path.join(stage_dir, "final.pt"), {"model": raw.state_dict(), "step": step,
                                                              "flops": flops, "model_cfg": cfg["model"]}, cfg)
        if args.stage == "plain":
            with open(os.path.join(stage_dir, "branch.json")) as f:
                br = json.load(f)
            with open(os.path.join(stage_dir, "budget.json"), "w") as f:
                json.dump({"total_flops": flops, "plain_steps": step, **br}, f, indent=2)
        if os.path.exists(last_path):
            os.remove(last_path)  # optimizer state no longer needed; save disk
        print(f"[{args.stage}] done: {step} steps, {flops:.3e} FLOPs, {(time.time() - t_start) / 60:.1f} min")
    barrier()
    cleanup_dist()


if __name__ == "__main__":
    main()
