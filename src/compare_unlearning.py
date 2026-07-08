"""
Compare de-memorization / unlearning methods on Pythia using the SAME WikiMIA
memorization panel as memorization_effect.py.

Each method is applied IN-PROCESS to a fresh copy of the model (DEL, SPE-Unlearn,
and the pruning baselines mutate weights directly — they are not saved
checkpoints), then scored with estimate_memorization so every method lands in one
comparison table alongside the baseline and, optionally, the gated-distillation
checkpoint (`--checkpoint trained_model`).

Data roles (matching train_model.py):
  forget set = WikiMIA members      (label == 1)   -> what we unlearn
  retain set = WikiMIA non-members  (label == 0)   -> utility to preserve
  eval set   = full WikiMIA split   (members + non-members) -> the MIA panel

DEL/SPE are restricted to the attention Q/K/V matrices — the same matrices the
pruning experiments and train_model.py's pruning step operate on — so the
comparison is on identical parameters.

The head-to-head is: our method (--checkpoint) vs DEL vs SPE. Our method is
members-only (ascent + KL); DEL and SPE additionally use the retain set (DEL's
finetune step, SPE's Fisher term). Pruning baselines remain in METHODS but are
not run by default.

Example:
  python src/compare_unlearning.py --length 64 \
      --methods del spe --checkpoint trained_model
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)   # memorization_effect, pruner
sys.path.insert(0, ROOT)   # methods package

from memorization_effect import (
    estimate_memorization, print_table, print_delta_table, calculate_perplexity,
)
from pruner import prune_l1_unstructured, prune_wanda, prune_global_l1_unstructured
from methods.del_unlearning import DELUnlearning
from methods.spe_unlearning import SPEUnlearning


# ── Data plumbing ─────────────────────────────────────────────────────────────

def attention_weight_names(model):
    """Fully-qualified `.weight` names of the attention Q/K/V projections.

    Mirrors the target set of pruner.py's pruning functions so DEL/SPE localize
    over exactly the same matrices.
    """
    names = []
    for name, module in model.named_modules():
        if hasattr(module, "query_key_value"):
            names.append(f"{name}.query_key_value.weight")
        elif hasattr(module, "q_proj"):
            names += [f"{name}.q_proj.weight", f"{name}.k_proj.weight", f"{name}.v_proj.weight"]
        elif hasattr(module, "c_attn"):
            names.append(f"{name}.c_attn.weight")
        elif hasattr(module, "attention") and hasattr(module.attention, "query"):
            names += [f"{name}.attention.query.weight", f"{name}.attention.key.weight",
                      f"{name}.attention.value.weight"]
    return names


def make_loader(texts, tokenizer, batch_size, max_length):
    """DataLoader of causal-LM batches: {input_ids, attention_mask, labels}.

    Pad positions are set to -100 in labels so they don't contribute to the loss.
    """
    pad_id = tokenizer.pad_token_id

    def collate(batch_texts):
        enc = tokenizer(batch_texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length)
        labels = enc["input_ids"].clone()
        if pad_id is not None:
            labels[labels == pad_id] = -100
        enc["labels"] = labels
        return dict(enc)

    return DataLoader(list(texts), batch_size=batch_size, shuffle=True, collate_fn=collate)


# ── Utility (general-capability) probe ────────────────────────────────────────

def load_utility_texts(n, min_chars=64):
    """Held-out wikitext-2 test lines — a corpus none of the methods train on.

    Serves as the clean utility signal: a faithful unlearning method should
    barely move perplexity here, while a method that merely broke the model
    will spike it.
    """
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t.strip() for t in wt["text"] if len(t.strip()) >= min_chars]
    return texts[:n]


def utility_perplexity(model, tokenizer, texts, max_length):
    """Mean held-out perplexity (lower = better; should stay near baseline)."""
    ppls = [calculate_perplexity(t, model, tokenizer, max_length=max_length) for t in texts]
    ppls = [p for p in ppls if np.isfinite(p)]
    return float(np.mean(ppls)) if ppls else float("nan")


# ── Method registry (each mutates `model` in place) ───────────────────────────

def apply_del(model, tokenizer, forget_loader, retain_loader, calib_texts, device, attn_names, args):
    d = DELUnlearning(model, device)
    crit = d.compute_criticality_scores(forget_loader, layer_names=attn_names, top_h=args.top_h)
    mask = d.generate_mask(crit, budget_alpha=args.budget_alpha)
    d.reset_parameters(mask)
    d.finetune_masked_params(retain_loader, mask, learning_rate=args.del_lr, epochs=args.epochs)


def apply_spe(model, tokenizer, forget_loader, retain_loader, calib_texts, device, attn_names, args):
    SPEUnlearning(model, device).unlearn(
        retain_loader, forget_loader,
        sparsity=args.sparsity, learning_rate=args.spe_lr, layer_names=attn_names,
    )


def apply_l1(model, tokenizer, forget_loader, retain_loader, calib_texts, device, attn_names, args):
    prune_l1_unstructured(model, prune_ratio=args.prune_ratio, layer_start=0)


def apply_wanda(model, tokenizer, forget_loader, retain_loader, calib_texts, device, attn_names, args):
    prune_wanda(model, tokenizer, calib_texts, prune_ratio=args.prune_ratio, device=device)


def apply_global_l1(model, tokenizer, forget_loader, retain_loader, calib_texts, device, attn_names, args):
    prune_global_l1_unstructured(model, prune_ratio=args.prune_ratio, layer_start=0)


METHODS = {
    "del": apply_del,
    "spe": apply_spe,
    "l1_unstructured": apply_l1,
    "wanda": apply_wanda,
    "global_l1": apply_global_l1,
}


# ── Driver ────────────────────────────────────────────────────────────────────

def load_model(name, device, dtype):
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-2.8b")
    ap.add_argument("--ref_model", default="EleutherAI/pythia-160m")
    ap.add_argument("--length", type=int, default=64, help="WikiMIA length split")
    # Focus comparison: our method (via --checkpoint) vs DEL vs SPE.
    # Pruning baselines stay available in METHODS but are not run by default.
    ap.add_argument("--methods", nargs="+", default=["del", "spe"],
                    choices=list(METHODS))
    ap.add_argument("--checkpoint", default=None,
                    help="Optional path to the gated-distillation checkpoint to include")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--budget_alpha", type=float, default=0.25, help="DEL update budget")
    ap.add_argument("--top_h", type=int, default=5, help="DEL channel top-h")
    ap.add_argument("--del_lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--sparsity", type=float, default=0.9, help="SPE freeze fraction")
    ap.add_argument("--spe_lr", type=float, default=1e-6)
    ap.add_argument("--prune_ratio", type=float, default=0.10)
    ap.add_argument("--n_utility", type=int, default=64,
                    help="# held-out wikitext-2 lines for the utility PPL probe")
    ap.add_argument("--fp32", action="store_true", help="Use float32 (default float16)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if args.fp32 else torch.float16
    print(f"Device: {device} | dtype: {dtype}")

    # Datasets
    full = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{args.length}")
    forget_texts = [ex["input"] for ex in full if ex["label"] == 1]
    retain_texts = [ex["input"] for ex in full if ex["label"] == 0]
    print(f"WikiMIA length{args.length}: {len(forget_texts)} members (forget), "
          f"{len(retain_texts)} non-members (retain)")

    # Held-out utility corpus (no method trains on it) — the general-capability probe.
    utility_texts = load_utility_texts(args.n_utility)
    print(f"Utility probe: {len(utility_texts)} held-out wikitext-2 lines")

    # Shared tokenizer + reference model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ref_tok = AutoTokenizer.from_pretrained(args.ref_model)
    if ref_tok.pad_token is None:
        ref_tok.pad_token = ref_tok.eos_token
    ref_model = load_model(args.ref_model, device, dtype).eval()

    results = {}
    utility = {}

    # Baseline
    print("\n=== Baseline ===")
    model = load_model(args.model, device, dtype).eval()
    results["Baseline"] = estimate_memorization(model, full, tokenizer, ref_model, ref_tok)
    utility["Baseline"] = utility_perplexity(model, tokenizer, utility_texts, args.max_length)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Each in-process method on a fresh model. A failure in one method is
    # isolated so the rest of the comparison still completes.
    failed = {}
    for key in args.methods:
        print(f"\n=== {key} ===")
        model = None
        try:
            model = load_model(args.model, device, dtype)
            attn_names = attention_weight_names(model)
            forget_loader = make_loader(forget_texts, tokenizer, args.batch_size, args.max_length)
            retain_loader = make_loader(retain_texts, tokenizer, args.batch_size, args.max_length)
            calib_texts = retain_texts[:32]  # Wanda calibration

            METHODS[key](model, tokenizer, forget_loader, retain_loader,
                         calib_texts, device, attn_names, args)

            model.eval()
            results[key] = estimate_memorization(model, full, tokenizer, ref_model, ref_tok)
            utility[key] = utility_perplexity(model, tokenizer, utility_texts, args.max_length)
        except Exception as e:
            failed[key] = f"{type(e).__name__}: {e}"
            print(f"  [skipped {key}] {failed[key]}")
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Optional gated-distillation checkpoint
    if args.checkpoint:
        print(f"\n=== checkpoint ({args.checkpoint}) ===")
        ck_tok = AutoTokenizer.from_pretrained(args.checkpoint)
        if ck_tok.pad_token is None:
            ck_tok.pad_token = ck_tok.eos_token
        model = load_model(args.checkpoint, device, dtype).eval()
        results["OurMethod"] = estimate_memorization(model, full, ck_tok, ref_model, ref_tok)
        utility["OurMethod"] = utility_perplexity(model, ck_tok, utility_texts, args.max_length)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Combined table + per-method deltas vs baseline
    print("\n" + "=" * 60)
    print("MEMORIZATION PANEL — all methods")
    print_table(results)

    baseline = results["Baseline"]
    for name, r in results.items():
        if name == "Baseline":
            continue
        print(f"\nDelta ({name} − Baseline)")
        print_delta_table(baseline, r)

    # Utility panel: held-out PPL a faithful method should leave near baseline.
    # A big positive Δ means the method broke general capability, not just forgot.
    print("\n" + "=" * 60)
    print("UTILITY — held-out wikitext-2 perplexity (lower = better)")
    base_ppl = utility.get("Baseline", float("nan"))
    print(f"\n{'Model':<12}{'PPL':>12}{'Δ vs base':>14}")
    print("-" * 38)
    for name, ppl in utility.items():
        delta = ppl - base_ppl
        tag = "" if name == "Baseline" else f"{delta:>+14.3f}"
        print(f"{name:<12}{ppl:>12.3f}{tag}")

    if failed:
        print("\nSkipped methods:")
        for key, msg in failed.items():
            print(f"  {key}: {msg}")


if __name__ == "__main__":
    main()
