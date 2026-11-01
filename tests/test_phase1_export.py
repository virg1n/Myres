"""Exported per-mode checkpoints must reproduce the training model (LoRA + FiLM code k) exactly.

    python3 -m tests.test_phase1_export      (CPU, builds a tiny random Qwen3; ~10 s)"""
import os
import tempfile

import torch

from phase1 import film as FM


def tiny_qwen(path):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.train_from_iterator(["hello world 123"] * 50, trainers.BpeTrainer(vocab_size=300, special_tokens=["<|im_end|>"],
                                                                          initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>").save_pretrained(path)
    cfg = Qwen3Config(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=1, head_dim=32, tie_word_embeddings=True)
    Qwen3ForCausalLM(cfg).save_pretrained(path)


def main():
    from transformers import AutoModelForCausalLM
    d = tempfile.mkdtemp()
    base = os.path.join(d, "base")
    tiny_qwen(base)
    model, film, ctx = FM.build(base, 4, 8, 16, "cpu", grad_ckpt=False)
    with torch.no_grad():
        for n, p in list(model.named_parameters()) + [("film." + n, p) for n, p in film.named_parameters()]:
            if n.endswith(".B") or n == "film.net.2.weight":
                p.normal_(0, 0.1)
    path = os.path.join(d, "adapter.pt")
    torch.save({"adapter": FM.adapter_state(model, film),
                "cfg": {"film_modes": 4, "lora_r": 8, "lora_alpha": 16}}, path)
    dirs = FM.export_modes(base, path, [os.path.join(d, f"m{k}") for k in range(4)], list(range(4)), dtype=torch.float32)
    x = torch.randint(0, 300, (2, 12))
    model.eval()
    for k in range(4):
        with torch.no_grad():
            ctx.gamma = film.gamma(torch.tensor([k, k]))
            a = model(input_ids=x).logits
            b = AutoModelForCausalLM.from_pretrained(dirs[k], dtype=torch.float32)(input_ids=x).logits
        diff = float((a - b).abs().max())
        assert diff < 1e-4, (k, diff)
    print("phase-1 export reproduces every mode: ok")


if __name__ == "__main__":
    main()
