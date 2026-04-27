"""
Three-part memorization experiment:
  1. Baseline  — original pythia-2.8b
  2. Global    — prune ALL layers at 5 / 10 / 15%
  3. Block-wise cumulative — for each ratio, prune one more layer per iteration
                             (layers 0..i) and evaluate after each step

Metrics reported per configuration:
  ROC-AUC (Min-K%)  |  Min-K member  |  Min-K non-member  |  Min-K gap
  PPL AUC  |  Lowercase AUC  |  Zlib AUC  |  Smaller-Ref AUC
"""

import json
import zlib

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pruner import prune_l1_unstructured

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME     = "EleutherAI/pythia-2.8b"
REF_MODEL_NAME = "EleutherAI/pythia-160m"
DATASET_LENGTH = 64
PRUNE_RATIOS   = [0.05, 0.10, 0.15]
OUTPUT_FILE    = "pruning_memorization_results.json"
MAX_LENGTH     = 2048


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model(name, device, dtype=torch.float16):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device)
    m.eval()
    return m, tok


def get_token_logprobs(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits[:, :-1, :]
        labels = inputs["input_ids"][:, 1:]
    lp = F.log_softmax(logits.float(), dim=-1)
    return lp.gather(-1, labels.unsqueeze(-1)).squeeze(-1).squeeze(0)


def min_k_score(token_log_probs, k=0.2):
    n = token_log_probs.numel()
    k_n = max(1, int(n * k))
    return torch.topk(token_log_probs, k_n, largest=False).values.mean().item()


def perplexity(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        loss = model(**inputs, labels=inputs["input_ids"]).loss
    return torch.exp(loss).item()


def safe_auc(scores, labels):
    scores = np.array(scores)
    labels = np.array(labels)
    if len(set(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


# ── Core evaluation ───────────────────────────────────────────────────────────

def evaluate(model, tokenizer, dataset, ref_model, ref_tokenizer, desc="eval"):
    mink, ppl_s, lower_s, zlib_s, ref_s = [], [], [], [], []
    labs = []

    for ex in tqdm(dataset, desc=f"  {desc}", leave=False):
        text  = ex["input"]
        label = ex["label"]

        p     = perplexity(text, model, tokenizer)
        p_low = perplexity(text.lower(), model, tokenizer)
        p_ref = perplexity(text, ref_model, ref_tokenizer)
        z     = len(zlib.compress(text.encode("utf-8")))
        lp    = get_token_logprobs(text, tokenizer, model)
        mk    = min_k_score(lp, k=0.2)

        mink.append(mk)
        ppl_s.append(-np.log(p))
        lower_s.append(np.log(p_low) / np.log(p))
        zlib_s.append(z / np.log(p))
        ref_s.append(np.log(p_ref) / np.log(p))
        labs.append(label)

    labs   = np.array(labs)
    mink_a = np.array(mink)
    mem    = mink_a[labs == 1]
    non    = mink_a[labs == 0]

    return {
        "roc_auc":          safe_auc(mink_a, labs),
        "mink_member":      float(mem.mean()) if len(mem) else 0.0,
        "mink_nonmember":   float(non.mean()) if len(non) else 0.0,
        "mink_gap":         float(mem.mean() - non.mean()) if len(mem) and len(non) else 0.0,
        "ppl_auc":          safe_auc(ppl_s,   labs),
        "lowercase_auc":    safe_auc(lower_s,  labs),
        "zlib_auc":         safe_auc(zlib_s,   labs),
        "smaller_ref_auc":  safe_auc(ref_s,    labs),
    }


# ── Table printer ─────────────────────────────────────────────────────────────

COLS = [
    ("ROC-AUC",  "roc_auc"),
    ("MK-Mem",   "mink_member"),
    ("MK-Non",   "mink_nonmember"),
    ("MK-Gap",   "mink_gap"),
    ("PPL",      "ppl_auc"),
    ("Lowercase","lowercase_auc"),
    ("Zlib",     "zlib_auc"),
    ("SmRef",    "smaller_ref_auc"),
]

def print_table(results):
    name_w = max(len(k) for k in results) + 2
    header = f"{'Experiment':<{name_w}}" + "".join(f"{h:>10}" for h, _ in COLS)
    sep    = "=" * len(header)
    print(f"\n{sep}\n{header}\n{'-'*len(header)}")
    for exp_name, r in results.items():
        row = f"{exp_name:<{name_w}}"
        for _, key in COLS:
            v = r.get(key)
            row += f"{v:>10.4f}" if v is not None else f"{'N/A':>10}"
        print(row)
    print(sep)


def save_incremental(results, path):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{DATASET_LENGTH}")
    n_mem = sum(1 for x in dataset if x["label"] == 1)
    n_non = sum(1 for x in dataset if x["label"] == 0)
    print(f"Dataset: {len(dataset)} samples  (members={n_mem}, non-members={n_non})")

    print(f"\nLoading reference model ({REF_MODEL_NAME})...")
    ref_model, ref_tok = load_model(REF_MODEL_NAME, device)

    num_layers = AutoConfig.from_pretrained(MODEL_NAME).num_hidden_layers
    print(f"Target model: {MODEL_NAME}  ({num_layers} layers)")

    all_results = {}

    # ── 1. Baseline ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PART 1: Baseline (no pruning)")
    print("="*60)
    model, tok = load_model(MODEL_NAME, device)
    all_results["Baseline"] = evaluate(model, tok, dataset, ref_model, ref_tok, "Baseline")
    print_table({"Baseline": all_results["Baseline"]})
    del model; torch.cuda.empty_cache()
    save_incremental(all_results, OUTPUT_FILE)

    # ── 2. Global pruning ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PART 2: Global pruning (all layers at once)")
    print("="*60)
    for ratio in PRUNE_RATIOS:
        pct = int(ratio * 100)
        key = f"Global {pct}%"
        print(f"\n[{key}] pruning all {num_layers} layers at {pct}%...")
        model, tok = load_model(MODEL_NAME, device)
        prune_l1_unstructured(model, prune_ratio=ratio, layer_start=0)
        all_results[key] = evaluate(model, tok, dataset, ref_model, ref_tok, key)
        print_table({key: all_results[key]})
        del model; torch.cuda.empty_cache()
        save_incremental(all_results, OUTPUT_FILE)

    # ── 3. Block-wise cumulative pruning ──────────────────────────────────────
    print("\n" + "="*60)
    print("PART 3: Block-wise cumulative pruning")
    print("  Load model once per ratio; prune one additional layer per step.")
    print("="*60)
    for ratio in PRUNE_RATIOS:
        pct = int(ratio * 100)
        print(f"\n  Ratio = {pct}%  — loading fresh model...")
        model, tok = load_model(MODEL_NAME, device)

        for layer_idx in range(num_layers):
            # Prune only this layer (incrementally accumulates 0..layer_idx)
            prune_l1_unstructured(model, prune_ratio=ratio,
                                  layer_start=layer_idx, layer_end=layer_idx)
            key = f"Block {pct}% L0-{layer_idx:02d}"
            print(f"\n  [{key}] layers 0–{layer_idx} pruned at {pct}%")
            all_results[key] = evaluate(model, tok, dataset, ref_model, ref_tok, key)
            print_table({key: all_results[key]})
            save_incremental(all_results, OUTPUT_FILE)

        del model; torch.cuda.empty_cache()

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n\n" + "="*60)
    print("FULL RESULTS SUMMARY")
    print("="*60)
    print_table(all_results)

    save_incremental(all_results, OUTPUT_FILE)
    print(f"\nAll results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
