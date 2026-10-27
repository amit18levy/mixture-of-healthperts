# MoE expert routing on medical text (Qwen3.6-35B-A3B × PubMedQA)

This project records which of the model's 256 experts each layer's router picks for every token of a PubMedQA
question, then compares expert usage across medical sub-domains.

## Files

| file | what it does |
|---|---|
| `common.py` | loads `rschf/pubmedqa`, tags each question with a sub-domain from its MeSH terms (`tags` column), samples a balanced set, writes the run directory and the per-domain summary |
| `moe_probe.cpp` | small llama.cpp program: registers an eval callback, grabs the `ffn_moe_topk-<layer>` (chosen expert ids) and `ffn_moe_probs-<layer>` (router probabilities) tensors, and sums them per prompt |
| `build.sh` | clones and builds llama.cpp (shared libs), then compiles `moe_probe` against it |
| `probe_llamacpp.py` | driver for GGUF models (CPU or GPU): runs `moe_probe`, checks that every token picked exactly `top_k` experts in every layer, and saves the run. It resumes if interrupted |
| `probe_hf.py` | the same analysis with Hugging Face `transformers` (forward hooks on each `mlp.gate`). Needs a GPU (about 70 GB in bf16, or about 20 GB with `--load-4bit`) |

## Run

```bash
pip install datasets numpy
./build.sh                          # for NVIDIA: CMAKE_EXTRA=-DGGML_CUDA=ON ./build.sh
python probe_llamacpp.py --model ../models/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf --per-domain 15 --out runs/gguf_iq2xxs
python probe_hf.py --per-domain 15 --out runs/hf_bf16          # GPU machine
```

To compare MoE configurations, run the same `--per-domain`/`--split` with different GGUF quantizations or with
the bf16 `transformers` model. The fixed seed picks the same questions each time, so the runs line up row by row.

## Output (one directory per run, the input for the UI)

- `counts.npy`: int32 `[n_samples, n_layers, n_experts]`, how many tokens picked each expert
- `probs.npy`: float32, same shape, summed router probability
- `samples.jsonl`: idx, PubMed id, domain, n_tokens, MeSH terms, question
- `run.json`: backend, model, quantization, top_k, layer ids, domain counts
- `summary.json`: per domain, `usage[layer][expert]` (normalized, for heatmaps) and `top_specialized`
  (the experts with the highest *lift*, meaning domain usage ÷ overall usage)

## Notes and caveats

- **Prefill only:** this measures routing of the input tokens (context and question), not of generated answers.
- **Last layer:** llama.cpp normally runs the last layer only for the final token, so `moe_probe` marks every token
  as an output to get full routing in layer 39.
- **Sub-domain labels:** these come from keyword rules (MeSH terms first, then the question), so they're approximate.
  "other" is the rest.
- **Low-bit quantization:** quantization below 4 bits can change routing. Check against a Q8/bf16 run before drawing strong conclusions.

## First results: `runs/gguf_iq2xxs_pqal150` (UD-IQ2_XXS, 150 pqal questions, 15 per sub-domain)

- **Experts in use:** about 254 of 256 experts per layer fire at least once, so there are no dead experts. Routing is close to uniform
  in the first layers (normalized entropy ~0.95) and more concentrated in middle and late layers (~0.8).
- **Domain vs. rest:** Jensen–Shannon distance between a domain's expert usage and everyone else's, compared with
  200 random shufflings of the labels (p99 = 0.152):
  psychiatry 0.214, infectious 0.161 and surgery 0.153 are above chance. The other sub-domains (0.125–0.143) can't be
  told apart from noise at n=15.
- **Clearest specialists:** psychiatry L15/E192 (25x the other domains' usage), L21/E87 (19x); infectious L8/E66 (14x);
  neurology L36/E203 (15x); pediatrics L20/E6 (13x); obstetrics L25/E229 (12x).
- **Layers 12 apart are linked.** In layers 8–19 and 20–31, expert *e* in layer *L* and expert *e* in layer *L+12*
  have very similar router weights (cosine 0.73–0.83, against ~0.32 for unrelated experts), and their usage is correlated
  (r = 0.39; every other layer gap is ~0). Layers 20–31 look like they were made by duplicating layers 8–19
  (depth up-scaling). So "the same expert" can show up twice, e.g. psychiatry E192 at L15 and L27.

These come from a 2-bit model with 15 questions per domain. Confirm them with more questions (`--split pqaa`) and a Q8/bf16 run.
