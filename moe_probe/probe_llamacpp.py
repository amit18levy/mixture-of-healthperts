#!/usr/bin/env python3
"""
llama.cpp probe (runs on CPU): record which experts a GGUF MoE model's routers pick on PubMedQA.

  ./build.sh                                   # compiles llama.cpp + the moe_probe binary
  python probe_llamacpp.py --model ../models/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf --per-domain 15 --out runs/gguf_iq2xxs

Rerunning with the same --out resumes: prompts already in routing.jsonl are skipped.
Writes the same run directory as probe_hf.py (see common.save_run).
"""
import argparse, json, re, subprocess
from pathlib import Path

import numpy as np

from common import load_samples, save_run

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to a .gguf file")
    ap.add_argument("--binary", default=str(HERE / "build" / "moe_probe"))
    ap.add_argument("--dataset", default="rschf/pubmedqa")
    ap.add_argument("--split", default="pqal", help="pqal (1k labeled), pqaa (211k), test (500)")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--per-domain", type=int, default=None, help="balanced sample: up to N per sub-domain")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=1024, help="max prompt tokens (longer prompts are truncated)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.dataset, args.split, args.n, args.per_domain)
    routing_path = out / "routing.jsonl"
    done = {json.loads(l)["idx"] for l in open(routing_path)} if routing_path.exists() else set()
    todo = [s for s in samples if s["idx"] not in done]
    print(f"{len(samples)} samples, {len(done)} already done, {len(todo)} to run")

    if todo:
        prompts = out / "prompts.jsonl"
        with open(prompts, "w") as f:
            for s in todo: f.write(json.dumps({"idx": s["idx"], "text": s["text"]}) + "\n")
        tmp = out / "routing.part.jsonl"
        try:
            subprocess.run([args.binary, "-m", args.model, "-i", str(prompts), "-o", str(tmp),
                            "-t", str(args.threads), "-c", str(args.ctx)], check=True)
        finally:  # keep whatever finished, so an interrupted run can resume
            if tmp.exists():
                with open(routing_path, "a") as f: f.write(tmp.read_text())
                tmp.unlink()

    routed = {}
    for line in open(routing_path):
        r = json.loads(line); routed[r["idx"]] = r
    samples = [s for s in samples if s["idx"] in routed]
    first = routed[samples[0]["idx"]]
    counts = np.stack([np.array(routed[s["idx"]]["counts"]) for s in samples])
    probs = np.stack([np.array(routed[s["idx"]]["probs"]) for s in samples])
    for s in samples:
        s["n_tokens"] = routed[s["idx"]]["n_tokens"]
    # Sanity check: every token must have picked exactly top_k experts in every layer.
    expected = np.array([s["n_tokens"] for s in samples])[:, None] * first["top_k"]
    assert (counts.sum(-1) == expected).all(), "expert counts != top_k * n_tokens"

    quant = re.sub(r"\.gguf$", "", Path(args.model).name).split("-")[-1]
    save_run(out, samples, counts, probs,
             {"backend": "llama.cpp", "model": Path(args.model).name, "quantization": quant,
              "top_k": first["top_k"], "layers": first["layers"], "dataset": args.dataset, "split": args.split})


if __name__ == "__main__":
    main()
