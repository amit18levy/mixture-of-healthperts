"""Shared pieces for the transformers and llama.cpp probes: data loading, sub-domain tagging,
balanced sampling, and writing a run directory in one format the UI can load."""
import json, random, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Keyword -> sub-domain. Matched against MeSH terms first (precise), then question text.
DOMAINS = {
    "oncology":    ["neoplasm", "cancer", "tumor", "tumour", "carcinoma", "lymphoma", "leukemia",
                    "metasta", "chemotherap", "oncolog", "sarcoma", "melanoma", "radiotherap"],
    "cardiology":  ["heart", "cardi", "coronary", "myocard", "atrial", "arrhythm", "hypertension",
                    "aort", "angina", "vascular"],
    "neurology":   ["brain", "neuro", "stroke", "epilep", "alzheimer", "parkinson", "dementia",
                    "cerebr", "spinal cord", "multiple sclerosis"],
    "infectious":  ["infect", "bacteri", "virus", "viral", "hiv", "sepsis", "antibiot",
                    "tubercul", "hepatitis", "vaccin"],
    "pediatrics":  ["infant", "pediatric", "paediatric", "neonat", "child"],
    "psychiatry":  ["depress", "anxiety", "psychiatr", "schizophren", "mental", "bipolar", "suicid"],
    "obstetrics":  ["pregnan", "obstet", "gynec", "maternal", "fetal", "fetus", "cesarean", "labor, obstetric"],
    "endocrine":   ["diabet", "insulin", "thyroid", "obesity", "glucose", "metabolic syndrome"],
    "surgery":     ["surg", "postoperative", "laparoscop", "anesthe", "transplant"],
}


def classify(meshes, question):
    """Pick the domain with the most keyword hits; MeSH hits weigh 2x more than question hits."""
    mesh_text, q = " | ".join(meshes).lower(), question.lower()
    scores = {d: 2 * sum(k in mesh_text for k in kws) + sum(k in q for k in kws)
              for d, kws in DOMAINS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "other"


def extract(row):
    """Tolerate both the original nested PubMedQA schema and flattened variants."""
    ctx = row.get("contexts") or row.get("context") or ""
    meshes = row.get("tags") or row.get("meshes") or []  # rschf/pubmedqa: `tags` = MeSH terms
    if isinstance(ctx, dict):
        meshes = meshes or ctx.get("meshes", [])
        ctx = ctx.get("contexts", "")
    if isinstance(ctx, list):
        ctx = " ".join(ctx)
    question = row.get("question") or row.get("QUESTION") or ""
    sid = row.get("pubid") or row.get("id")
    return sid, question, ctx, list(meshes or [])


def js_distance(p, q):
    """Jensen-Shannon distance (base 2, in [0, 1]) along the last axis."""
    m = (p + q) / 2
    kl = lambda a, b: (a * np.log((a + 1e-12) / (b + 1e-12))).sum(-1)
    return np.sqrt(np.clip((kl(p, m) + kl(q, m)) / 2 / np.log(2), 0, None))


def summarize(counts, domains, top=10, n_perm=200, seed=0):
    """Per domain: expert-usage distribution, how far it is from the other domains (vs. a shuffled-label
    baseline), and the experts it uses most relative to the rest. Each sample is normalized to a per-layer
    distribution first so long prompts don't dominate. A specialist must be used in >= 2/3 of the domain's
    samples at >= 2x the uniform share, so rare experts that fired once don't get huge lifts."""
    F = counts / np.maximum(counts.sum(-1, keepdims=True), 1)  # [S, L, E]
    dom = np.array(domains)
    E = F.shape[-1]
    dist = lambda lab, d: js_distance(F[lab == d].mean(0), F[lab != d].mean(0))  # [L]
    rng = np.random.default_rng(seed)
    null = np.array([[dist(p, d).mean() for d in np.unique(dom)]
                     for p in (rng.permutation(dom) for _ in range(n_perm))])
    out = {}
    for d in np.unique(dom):
        m = dom == d
        dm, rm = F[m].mean(0), F[~m].mean(0)
        support = (F[m] > 0).mean(0)
        lift = np.where((dm >= 2 / E) & (support >= 2 / 3), dm / (rm + 1e-6), 0)
        flat = [i for i in np.argsort(lift, axis=None)[::-1][:top] if lift.flat[i] > 0]
        js_layers = dist(dom, d)
        out[str(d)] = {
            "n_samples": int(m.sum()),
            "usage": dm.round(6).tolist(),  # [L][E], for heatmaps
            "js_vs_rest": round(float(js_layers.mean()), 4),
            "js_vs_rest_by_layer": js_layers.round(4).tolist(),
            "js_null_mean": round(float(null.mean()), 4),
            "js_null_p99": round(float(np.percentile(null, 99)), 4),
            "top_specialized": [{"layer": int(l), "expert": int(e), "lift": round(float(lift[l, e]), 2),
                                 "freq": round(float(dm[l, e]), 5), "support": round(float(support[l, e]), 2)}
                                for l, e in zip(*np.unravel_index(flat, lift.shape))],
        }
    return out


def load_samples(dataset="rschf/pubmedqa", split="pqal", n=None, per_domain=None, seed=0):
    """Return [{idx, id, question, text, meshes, domain}], optionally balanced to `per_domain` each."""
    from datasets import load_dataset
    rows = []
    for i, row in enumerate(load_dataset(dataset, split=split)):
        sid, q, ctx, meshes = extract(row)
        rows.append({"idx": i, "id": sid, "question": q, "meshes": meshes,
                     "domain": classify(meshes, q), "text": f"Context: {ctx}\nQuestion: {q}\nAnswer:"})
    if per_domain:
        by_dom = defaultdict(list)
        for r in rows: by_dom[r["domain"]].append(r)
        rng = random.Random(seed)
        rows = sorted((r for d in by_dom.values() for r in rng.sample(d, min(per_domain, len(d)))),
                      key=lambda r: r["idx"])
    return rows[:n] if n else rows


def save_run(out, samples, counts, probs, meta):
    """samples: list of dicts (with domain, n_tokens); counts/probs: [n_samples, n_layers, n_experts]."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "counts.npy", counts.astype(np.int32)); np.save(out / "probs.npy", probs.astype(np.float32))
    with open(out / "samples.jsonl", "w") as f:
        for s in samples:
            f.write(json.dumps({k: s[k] for k in ("idx", "id", "domain", "n_tokens", "meshes", "question")}) + "\n")
    domains = [s["domain"] for s in samples]
    meta = {**meta, "n_layers": counts.shape[1], "n_experts": counts.shape[2], "n_samples": len(samples),
            "domains": dict(Counter(domains)), "created": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(meta, open(out / "run.json", "w"), indent=2)
    summary = summarize(counts, domains)
    json.dump(summary, open(out / "summary.json", "w"))
    for d, s in summary.items():
        best = ", ".join(f"L{x['layer']}/E{x['expert']}(x{x['lift']})" for x in s["top_specialized"][:5])
        sig = "*" if s["js_vs_rest"] > s["js_null_p99"] else " "
        print(f"{d:<11} n={s['n_samples']:<4} JS={s['js_vs_rest']:.3f}{sig} (null p99 {s['js_null_p99']:.3f})  specialists: {best}")
