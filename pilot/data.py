"""On-the-fly training data: nothing is written to disk.

Each DataLoader worker generates problems, renders solve / verify sequences, sorts a pool by
length (little padding) and yields ready micro-batches."""
import random

import torch
from torch.utils.data import IterableDataset, get_worker_info

from . import igsm


def make_example(rng, op_lo, op_hi, verify_frac, max_len):
    prob = igsm.sample_problem(rng, "train", op_lo, op_hi, max_len)
    if rng.random() < verify_frac:
        ids, loss_from = igsm.verify_sequence(prob, rng, with_error=rng.random() < 0.5)
        return ids, loss_from, True
    ids, loss_from = igsm.solve_sequence(prob, rng)
    return ids, loss_from, False


def collate(examples, pad_to=64):
    """examples: list of (ids, loss_from, is_verify[, weight]).

    Targets are -100 before loss_from and on padding."""
    T = max(len(e[0]) for e in examples) - 1
    T = ((T + pad_to - 1) // pad_to) * pad_to
    B = len(examples)
    x = torch.full((B, T), igsm.PAD, dtype=torch.long)
    y = torch.full((B, T), -100, dtype=torch.long)
    for i, e in enumerate(examples):
        ids, lf = e[0], e[1]
        n = len(ids) - 1
        x[i, :n] = torch.tensor(ids[:-1])
        tgt = torch.tensor(ids[1:])
        tgt[:max(lf - 1, 0)] = -100
        y[i, :n] = tgt
    ver = torch.tensor([bool(e[2]) for e in examples])
    w = torch.tensor([float(e[3]) if len(e) > 3 else 1.0 for e in examples])
    ntok = sum(len(e[0]) for e in examples)
    return {"x": x, "y": y, "verify": ver, "w": w, "ntok": ntok}


class TrainStream(IterableDataset):
    def __init__(self, seed, op_lo, op_hi, verify_frac, micro_bs, max_len=1024, pool=8):
        self.seed, self.op_lo, self.op_hi = seed, op_lo, op_hi
        self.verify_frac, self.micro_bs, self.max_len, self.pool = verify_frac, micro_bs, max_len, pool

    def __iter__(self):
        wi = get_worker_info()
        wid = wi.id if wi else 0
        rng = random.Random(self.seed * 1000 + wid)
        while True:
            exs = [make_example(rng, self.op_lo, self.op_hi, self.verify_frac, self.max_len)
                   for _ in range(self.micro_bs * self.pool)]
            exs.sort(key=lambda e: len(e[0]))
            batches = [exs[i:i + self.micro_bs] for i in range(0, len(exs), self.micro_bs)]
            rng.shuffle(batches)
            for b in batches:
                yield collate(b)


def train_loader(seed, op_lo, op_hi, verify_frac, micro_bs, workers, max_len=1024):
    ds = TrainStream(seed, op_lo, op_hi, verify_frac, micro_bs, max_len)
    if workers == 0:
        return iter(ds)
    dl = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=workers, pin_memory=True,
                                     prefetch_factor=4, persistent_workers=True)
    return iter(dl)
