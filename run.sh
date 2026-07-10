#!/bin/bash
# End-to-end pipeline: train our gated-distillation method, then compare it
# head-to-head against the frozen DEL / SPE unlearning baselines.
set -euo pipefail

# ── Shared config ─────────────────────────────────────────────────────────────
# Train and evaluate on the SAME WikiMIA split, or you forget one panel and
# score another.
LENGTH=64
MODEL="EleutherAI/pythia-2.8b"
CHECKPOINT="trained_model"

# Our method's utility-vs-forgetting dial. >0 anchors utility via the KL penalty.
KL_WEIGHT=0.1
TRAIN_EPOCHS=1
TRAIN_LR=1e-5

# Frozen baseline configs (from src/sweep_unlearning.py, budget = 1.5x baseline PPL):
#   DEL: alpha=0.005, epochs=3   -> AUC 0.461, PPL 51.0
#   SPE: lr=1e-7                 -> AUC 0.516, PPL 32.6
DEL_ALPHA=0.005
DEL_EPOCHS=3
SPE_LR=1e-7

# ── 1. Execute our method (produces $CHECKPOINT) ──────────────────────────────
echo "=== Training our method -> ${CHECKPOINT} ==="
python3 src/train_model.py \
    --model "${MODEL}" \
    --save_path "${CHECKPOINT}" \
    --length "${LENGTH}" \
    --epochs "${TRAIN_EPOCHS}" \
    --lr "${TRAIN_LR}" \
    --kl_weight "${KL_WEIGHT}"

# Pruning vs. memorization analysis for our method.
echo "=== Pruning/memorization experiment ==="
python3 src/pruning_memorization_experiment.py

# ── 2. Head-to-head comparison (our method vs. DEL vs. SPE vs. baseline) ───────
echo "=== Comparison: ours vs DEL vs SPE ==="
python3 src/compare_unlearning.py \
    --length "${LENGTH}" \
    --fp32 \
    --methods del spe \
    --budget_alpha "${DEL_ALPHA}" \
    --epochs "${DEL_EPOCHS}" \
    --spe_lr "${SPE_LR}"

echo "=== Pipeline complete ==="
