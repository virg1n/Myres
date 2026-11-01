"""Inference backends: vLLM (fast, default) and HF transformers (fallback, slower).

Both expose:
  generate(prompts, n, temperature, top_p, top_k, max_tokens) -> list[list[str]]
  p_no(prompts) -> (list[float] P(No) among {Yes, No}, n_missing)
"""
import gc
import math
import os
import shutil

YES = ["Yes", " Yes", "yes", " yes", "YES"]
NO = ["No", " No", "no", " no", "NO"]


def single_token_ids(tok, words):
    ids = set()
    for w in words:
        t = tok.encode(w, add_special_tokens=False)
        if len(t) == 1:
            ids.add(t[0])
    return sorted(ids)


class VLLMBackend:
    def __init__(self, path, seed=0, max_len=4096, gpu_mem=0.85, eager=False):
        from transformers import AutoTokenizer
        # FlashInfer's top-k/top-p sampler is JIT-compiled and needs nvcc; the PyTorch sampler does not
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import LLM
        self.llm = LLM(model=path, dtype="bfloat16", seed=seed, max_model_len=max_len,
                       gpu_memory_utilization=gpu_mem, enforce_eager=eager, enable_prefix_caching=True)
        self.tok = AutoTokenizer.from_pretrained(path)    # prompts are built with the HF chat template
        self.yes, self.no = single_token_ids(self.tok, YES), single_token_ids(self.tok, NO)

    def generate(self, prompts, n, temperature, top_p, top_k, max_tokens):
        from vllm import SamplingParams
        sp = SamplingParams(n=n, temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=max_tokens)
        outs = self.llm.generate(prompts, sp, use_tqdm=False)
        return [[o.text for o in r.outputs] for r in outs]

    def p_no(self, prompts):
        from vllm import SamplingParams
        sp = SamplingParams(n=1, temperature=0.0, max_tokens=1, logprobs=20)
        outs = self.llm.generate(prompts, sp, use_tqdm=False)
        res, missing = [], 0
        for r in outs:
            lp = r.outputs[0].logprobs[0] if r.outputs[0].logprobs else {}
            py = sum(math.exp(lp[i].logprob) for i in self.yes if i in lp)
            pn = sum(math.exp(lp[i].logprob) for i in self.no if i in lp)
            if py + pn == 0:
                missing += 1
                res.append(0.5)
            else:
                res.append(pn / (py + pn))
        return res, missing


class HFBackend:
    def __init__(self, path, seed=0, max_len=4096, batch_size=64, **_):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.dev.type == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(path)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(self.dev).eval()
        self.bs, self.max_len = batch_size, max_len
        torch.manual_seed(seed)
        self.yes, self.no = single_token_ids(self.tok, YES), single_token_ids(self.tok, NO)

    def _batches(self, prompts, bs):
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        for s in range(0, len(order), bs):
            yield order[s:s + bs]

    def generate(self, prompts, n, temperature, top_p, top_k, max_tokens):
        torch = self.torch
        out = [None] * len(prompts)
        bs = max(1, self.bs // n)
        for idx in self._batches(prompts, bs):
            enc = self.tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(self.dev)
            kw = dict(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k) if temperature > 0 \
                else dict(do_sample=False)
            with torch.no_grad():
                g = self.model.generate(**enc, max_new_tokens=max_tokens, num_return_sequences=n,
                                        pad_token_id=self.tok.pad_token_id, **kw)
            new = g[:, enc["input_ids"].shape[1]:]
            texts = self.tok.batch_decode(new, skip_special_tokens=True)
            for r, i in enumerate(idx):
                out[i] = texts[r * n:(r + 1) * n]
        return out

    def p_no(self, prompts):
        torch = self.torch
        res = [0.5] * len(prompts)
        for idx in self._batches(prompts, self.bs):
            enc = self.tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(self.dev)
            with torch.no_grad():
                logits = self.model(**enc).logits[:, -1].float()
            p = torch.softmax(logits, -1)
            py, pn = p[:, self.yes].sum(-1), p[:, self.no].sum(-1)
            for r, i in enumerate(idx):
                res[i] = float(pn[r] / (py[r] + pn[r])) if float(py[r] + pn[r]) > 0 else 0.5
        return res, 0


def load_backend(kind, path, **kw):
    if kind == "auto":
        try:
            import vllm  # noqa: F401
            kind = "vllm"
        except Exception:
            kind = "hf"
    if kind == "vllm":
        return VLLMBackend(path, **kw)
    kw.pop("eager", None)
    kw.pop("gpu_mem", None)
    return HFBackend(path, **kw)


def make_noisy(base_path, out_dir, sigma, seed):
    """Write a copy of the model with W += sigma * std(W) * N(0,1) on every 2-D weight except
    embeddings / lm_head. Done on CPU, one tensor at a time; ~3.5 GB on disk for 1.7B."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    model = AutoModelForCausalLM.from_pretrained(base_path, dtype=torch.bfloat16)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() != 2 or "embed" in name or "lm_head" in name:
                continue
            w = p.float()
            w.add_(torch.randn(w.shape, generator=g) * (sigma * w.std()))
            p.copy_(w.to(p.dtype))
            del w
    model.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(base_path).save_pretrained(out_dir)
    del model
    gc.collect()
    return out_dir


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make_noisy"])
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    make_noisy(a.base, a.out, a.sigma, a.seed)
    print("noisy model written to", a.out, flush=True)
