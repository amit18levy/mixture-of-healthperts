#!/usr/bin/env python3
"""
Transformers probe (needs a GPU): record which experts Qwen3.6-35B-A3B's routers pick on PubMedQA.

  python probe_hf.py --per-domain 50 --out runs/hf_bf16
  python probe_hf.py --per-domain 50 --load-4bit --out runs/hf_nf4

Writes the same run directory as probe_llamacpp.py (see common.save_run).
"""
import argparse, re

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import load_samples, save_run


class RouterRecorder:
    """Forward hooks on every `layers.N.mlp.gate` (the router). Works whether the gate returns raw
    logits (nn.Linear) or a (logits/probs, ...) tuple, since softmax preserves top-k order."""

    def __init__(self, model, top_k):
        self.top_k, self.buf, self.handles = top_k, {}, []
        pat = re.compile(r"layers\.(\d+)\.mlp\.gate$")
        for name, mod in model.named_modules():
            if m := pat.search(name):
                self.handles.append(mod.register_forward_hook(self._hook(int(m.group(1)))))
        if not self.handles:
            raise RuntimeError("No router modules found; print(model) and adjust the regex.")
        self.n_layers = len(self.handles)

    def _hook(self, layer):
        def fn(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            self.buf[layer] = t.detach().float().reshape(-1, t.shape[-1])  # [tokens, n_experts]
        return fn

    def pop(self):
        """Return (counts, probs) arrays [n_layers, n_experts] for the last forward pass."""
        layers = sorted(self.buf)
        n_exp = self.buf[layers[0]].shape[-1]
        counts = np.zeros((len(layers), n_exp), np.int32)
        probs = np.zeros((len(layers), n_exp), np.float32)
        for i, l in enumerate(layers):
            x = self.buf[l]
            p = x if torch.allclose(x.sum(-1), torch.ones(1, device=x.device), atol=1e-3) else x.softmax(-1)
            idx = p.topk(self.top_k, dim=-1).indices
            counts[i] = torch.bincount(idx.flatten(), minlength=n_exp).cpu().numpy()
            probs[i] = p.sum(0).cpu().numpy()
        self.buf.clear()
        return counts, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--dataset", default="rschf/pubmedqa")
    ap.add_argument("--split", default="pqal", help="pqal (1k labeled), pqaa (211k), test (500)")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--per-domain", type=int, default=None, help="balanced sample: up to N per sub-domain")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--load-4bit", action="store_true", help="bitsandbytes NF4, to fit on a single ~24GB GPU")
    ap.add_argument("--out", default="runs/hf_bf16")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    kw = dict(device_map="auto", torch_dtype=torch.bfloat16)
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw).eval()
    cfg = getattr(model.config, "text_config", model.config)
    rec = RouterRecorder(model, cfg.num_experts_per_tok)

    samples = load_samples(args.dataset, args.split, args.n, args.per_domain)
    all_counts, all_probs = [], []
    for i, s in enumerate(samples):
        ids = tok(s["text"], return_tensors="pt", truncation=True, max_length=args.max_tokens).to(model.device)
        with torch.inference_mode():
            model(**ids, use_cache=False)  # prefill only; routing of the input tokens
        c, p = rec.pop()
        all_counts.append(c); all_probs.append(p); s["n_tokens"] = ids.input_ids.shape[1]
        if i % 25 == 0:
            print(f"[{i}/{len(samples)}] {s['domain']:<11} {s['n_tokens']} tok")

    save_run(args.out, samples, np.stack(all_counts), np.stack(all_probs),
             {"backend": "transformers", "model": args.model, "quantization": "nf4" if args.load_4bit else "bf16",
              "top_k": cfg.num_experts_per_tok, "dataset": args.dataset, "split": args.split})


if __name__ == "__main__":
    main()
