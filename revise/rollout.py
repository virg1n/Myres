"""Plain left-to-right generation with a KV cache (writer stream only), used
for (a) the plain baseline at evaluation time and (b) on-policy rollouts during
revision training: the model's own mistakes are collected into a replay buffer.
"""

import collections
import random

import torch

from .igsm.checker import analyze
from .model import KVCache


@torch.no_grad()
def generate(model, prompts, max_new, eos, pad, device, temperature=0.0, amp_dtype=None, generator=None):
    """Batched sampling. prompts: list of token lists. Returns list of generated
    token lists (each ends with eos unless max_new was hit). Also returns the
    number of tokens processed (for compute accounting)."""
    model.eval()
    B = len(prompts)
    P = max(len(p) for p in prompts)
    ids = torch.full((B, P), pad, dtype=torch.long)
    valid = torch.zeros((B, P + max_new), dtype=torch.bool)
    pos = torch.zeros((B, P), dtype=torch.long)
    for b, p in enumerate(prompts):
        n = len(p)
        ids[b, P - n:] = torch.tensor(p)
        pos[b, P - n:] = torch.arange(n)
        valid[b, P - n:P] = True
    ids, valid, pos = ids.to(device), valid.to(device), pos.to(device)
    dtype = amp_dtype if (amp_dtype is not None and device.type == "cuda") else torch.float32
    cache = KVCache(model.cfg, B, P + max_new, device, dtype)
    stream = torch.zeros_like(ids)
    causal = torch.tril(torch.ones(P, P, dtype=torch.bool, device=device))
    mask = (causal.unsqueeze(0) & valid[:, None, :P]).unsqueeze(1)
    mask = mask | torch.eye(P, dtype=torch.bool, device=device)[None, None]
    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None and device.type == "cuda"):
        h = model(ids, pos, stream, mask, cache=cache)
        logits = model.lm_logits(h[:, -1]).float()
    out = [[] for _ in range(B)]
    done = torch.zeros(B, dtype=torch.bool, device=device)
    cur_pos = pos[:, -1]
    processed = sum(len(p) for p in prompts)
    for step in range(max_new):
        if temperature and temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator).squeeze(1)
        else:
            nxt = logits.argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, pad), nxt)
        nl = nxt.tolist()
        dl = done.tolist()
        for b in range(B):
            if not dl[b]:
                out[b].append(nl[b])
        processed += int((~done).sum())
        done = done | (nxt == eos)
        if bool(done.all()) or step == max_new - 1:
            break
        valid[:, P + step] = True
        cur_pos = cur_pos + 1
        m = valid[:, None, None, : P + step + 1]
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None and device.type == "cuda"):
            h = model(nxt.unsqueeze(1), cur_pos.unsqueeze(1), torch.zeros_like(nxt).unsqueeze(1), m, cache=cache)
            logits = model.lm_logits(h[:, -1]).float()
    return out, processed


class RolloutBuffer:
    """Replay buffer of the model's own erroneous solutions."""

    def __init__(self, maxlen, seed=0):
        self.items = collections.deque(maxlen=maxlen)
        self.rng = random.Random(seed)
        self.stats = collections.deque(maxlen=50)

    def __len__(self):
        return len(self.items)

    def add(self, e):
        self.items.append(e)

    def sample(self, n):
        if not self.items:
            return []
        return [self.items[self.rng.randrange(len(self.items))] for _ in range(n)]


def collect_rollouts(model, sampler, tok, n, device, temps, max_new, amp_dtype, batch_size, op_range, rng,
                     strict_necessary=True):
    """Generate n solutions with the current model, check them, return
    (error entries, stats, tokens processed)."""
    entries, n_correct, n_err, processed = [], 0, 0, 0
    for s in range(0, n, batch_size):
        m = min(batch_size, n - s)
        probs = [sampler(op_range) for _ in range(m)]
        temp = rng.choice(temps)
        gens, proc = generate(model, [p.prompt_ids for p in probs], max_new, tok.eos, tok.pad, device,
                              temperature=temp, amp_dtype=amp_dtype)
        processed += proc
        for p, g in zip(probs, gens):
            a = analyze(p, g, tok, strict_necessary)
            n_correct += int(a.correct)
            if a.first_err is not None:
                n_err += 1
                entries.append({"prompt": p.prompt_ids, "sol": g, "status": a.status, "valid": a.valid,
                                "parents": a.parents, "j": a.first_err, "op": p.n_ops, "temp": temp})
    return entries, {"rollout_acc": n_correct / max(1, n), "rollout_err_rate": n_err / max(1, n)}, processed
