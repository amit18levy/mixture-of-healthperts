# mixture-of-healthperts
Interpreting Mixture-of-Experts in HealthCare

## What's here

- [`moe_probe/`](moe_probe/): records which experts Qwen3.6-35B-A3B's routers pick on PubMedQA questions, split by
  medical sub-domain (oncology, cardiology, neurology, ...). It works with llama.cpp (GGUF, CPU or GPU) and with
  Hugging Face `transformers` (GPU). See [`moe_probe/README.md`](moe_probe/README.md) for how to run it and for the first results.
