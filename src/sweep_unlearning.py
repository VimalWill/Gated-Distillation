"""
Sweep DEL and SPE over their key knobs and report, for each config, the MIA
ROC-AUC and held-out utility perplexity — the two numbers you freeze a baseline
on.

The goal is to pick each method's fair "frozen" config with a single rule:
    strongest forgetting (AUC closest to 0.5) SUBJECT TO a utility budget
    (held-out PPL <= budget_mult * baseline PPL).

This reuses the exact scoring/data machinery from compare_unlearning.py so the
sweep and the head-to-head comparison land on identical metrics.

Each config reloads a fresh copy of the base model (the methods mutate weights
in place), so runtime is roughly (#configs + 1) full model loads.

Example:
  python src/sweep_unlearning.py --length 64 --fp32 --budget_mult 1.5
"""

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from memorization_effect import estimate_memorization
from compare_unlearning import (
    attention_weight_names, make_loader, load_utility_texts, utility_perplexity,
    load_model, apply_del, apply_spe,
)
from datasets import load_dataset
from transformers import AutoTokenizer


# ── Sweep grids (edit or override via CLI) ────────────────────────────────────
# DEL: (budget_alpha, epochs) — reset budget vs. recovery finetuning.
DEL_GRID = [
    (0.02, 3),
    (0.05, 2),
    (0.05, 3),
    (0.10, 2),
]
# SPE: learning rates at the default damping. lr sets the Newton step scale.
SPE_LRS = [1e-7, 3e-7, 1e-6, 3e-6]


def base_args(cli):
    """Shared method knobs held fixed across the sweep (per-config knobs overridden)."""
    return dict(
        top_h=cli.top_h, del_lr=cli.del_lr,
        sparsity=cli.sparsity, spe_damping=cli.spe_damping,
        spe_max_update=cli.spe_max_update, max_length=cli.max_length,
    )


def run_one(apply_fn, cfg, base_model_name, tokenizer, full, ref_model, ref_tok,
            forget_texts, retain_texts, utility_texts, device, dtype, cli):
    """Load a fresh model, apply one method config, return (auc, ppl)."""
    model = load_model(base_model_name, device, dtype)
    try:
        attn_names = attention_weight_names(model)
        forget_loader = make_loader(forget_texts, tokenizer, cli.batch_size, cli.max_length)
        retain_loader = make_loader(retain_texts, tokenizer, cli.batch_size, cli.max_length)
        args = SimpleNamespace(**{**base_args(cli), **cfg})
        apply_fn(model, tokenizer, forget_loader, retain_loader,
                 retain_texts[:32], device, attn_names, args)
        model.eval()
        res = estimate_memorization(model, full, tokenizer, ref_model, ref_tok)
        ppl = utility_perplexity(model, tokenizer, utility_texts, cli.max_length)
        return res["roc_auc"], ppl
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-2.8b")
    ap.add_argument("--ref_model", default="EleutherAI/pythia-160m")
    ap.add_argument("--length", type=int, default=64)
    ap.add_argument("--methods", nargs="+", default=["del", "spe"], choices=["del", "spe"])
    ap.add_argument("--budget_mult", type=float, default=1.5,
                    help="Utility budget: held-out PPL <= budget_mult * baseline PPL")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--n_utility", type=int, default=64)
    # Fixed method knobs
    ap.add_argument("--top_h", type=int, default=5)
    ap.add_argument("--del_lr", type=float, default=5e-5)
    ap.add_argument("--sparsity", type=float, default=0.9)
    ap.add_argument("--spe_damping", type=float, default=1e-2)
    ap.add_argument("--spe_max_update", type=float, default=1.0)
    # Grid overrides
    ap.add_argument("--del_epochs", type=int, nargs="+", default=None,
                    help="Override DEL epochs grid (paired with --del_alphas or default alpha 0.05)")
    ap.add_argument("--del_alphas", type=float, nargs="+", default=None,
                    help="Override DEL budget_alpha grid (cartesian with --del_epochs)")
    ap.add_argument("--spe_lrs", type=float, nargs="+", default=None,
                    help="Override SPE lr grid")
    ap.add_argument("--fp32", action="store_true")
    cli = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if cli.fp32 else torch.float16
    print(f"Device: {device} | dtype: {dtype}")

    # Data
    full = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{cli.length}")
    forget_texts = [ex["input"] for ex in full if ex["label"] == 1]
    retain_texts = [ex["input"] for ex in full if ex["label"] == 0]
    utility_texts = load_utility_texts(cli.n_utility)
    print(f"WikiMIA length{cli.length}: {len(forget_texts)} forget / {len(retain_texts)} retain; "
          f"{len(utility_texts)} utility lines")

    tokenizer = AutoTokenizer.from_pretrained(cli.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ref_tok = AutoTokenizer.from_pretrained(cli.ref_model)
    if ref_tok.pad_token is None:
        ref_tok.pad_token = ref_tok.eos_token
    ref_model = load_model(cli.ref_model, device, dtype).eval()

    # Baseline
    print("\n=== Baseline ===")
    model = load_model(cli.model, device, dtype).eval()
    base = estimate_memorization(model, full, tokenizer, ref_model, ref_tok)
    base_ppl = utility_perplexity(model, tokenizer, utility_texts, cli.max_length)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    budget = cli.budget_mult * base_ppl
    print(f"Baseline: AUC={base['roc_auc']:.4f}  PPL={base_ppl:.3f}  "
          f"-> utility budget PPL <= {budget:.3f} ({cli.budget_mult}x)")

    # Build grids
    del_grid = DEL_GRID
    if cli.del_epochs or cli.del_alphas:
        alphas = cli.del_alphas or [0.05]
        epochs = cli.del_epochs or [3]
        del_grid = [(a, e) for a in alphas for e in epochs]
    spe_lrs = cli.spe_lrs or SPE_LRS

    rows = []  # (method, label, auc, ppl)
    if "del" in cli.methods:
        for alpha, ep in del_grid:
            label = f"a={alpha},ep={ep}"
            print(f"\n=== del [{label}] ===")
            auc, ppl = run_one(apply_del, dict(budget_alpha=alpha, epochs=ep),
                               cli.model, tokenizer, full, ref_model, ref_tok,
                               forget_texts, retain_texts, utility_texts, device, dtype, cli)
            rows.append(("del", label, auc, ppl))
    if "spe" in cli.methods:
        for lr in spe_lrs:
            label = f"lr={lr:g}"
            print(f"\n=== spe [{label}] ===")
            auc, ppl = run_one(apply_spe, dict(spe_lr=lr),
                               cli.model, tokenizer, full, ref_model, ref_tok,
                               forget_texts, retain_texts, utility_texts, device, dtype, cli)
            rows.append(("spe", label, auc, ppl))

    # ── Report ────────────────────────────────────────────────────────────────
    def fmt_ppl(x):
        return f"{x:.3f}" if abs(x) < 1e6 else f"{x:.3e}"

    print("\n" + "=" * 74)
    print(f"SWEEP — target AUC≈0.5, utility budget PPL <= {budget:.1f}")
    print("=" * 74)
    print(f"{'Method':<7}{'Config':<16}{'AUC':>9}{'|AUC-0.5|':>11}{'PPL':>12}{'InBudget':>10}")
    print("-" * 74)
    for method, label, auc, ppl in rows:
        in_budget = np.isfinite(ppl) and ppl <= budget
        print(f"{method:<7}{label:<16}{auc:>9.4f}{abs(auc-0.5):>11.4f}"
              f"{fmt_ppl(ppl):>12}{('yes' if in_budget else 'no'):>10}")

    # Frozen pick per method: within budget, AUC closest to 0.5.
    print("\nFROZEN PICK (within budget, AUC closest to 0.5):")
    for method in cli.methods:
        cands = [(label, auc, ppl) for m, label, auc, ppl in rows
                 if m == method and np.isfinite(ppl) and ppl <= budget]
        if not cands:
            print(f"  {method}: no config meets the budget — "
                  f"loosen --budget_mult or extend the grid")
            continue
        label, auc, ppl = min(cands, key=lambda c: abs(c[1] - 0.5))
        print(f"  {method}: {label}  (AUC={auc:.4f}, PPL={fmt_ppl(ppl)})")


if __name__ == "__main__":
    main()
