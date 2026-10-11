#!/usr/bin/env bash
# Run inside an active HPC GPU allocation, from an activated Python environment.
set -euo pipefail
cd "$(dirname "$0")"
RUN_DIR="${1:?Usage: bash train_pythia_full_lm.sh NEW_RUN_DIRECTORY}"
if [[ -e "$RUN_DIR" ]]; then
  echo "Choose a fresh run directory: $RUN_DIR" >&2
  exit 1
fi
python3 -c 'import torch, transformers, datasets, sklearn, lm_eval; assert torch.cuda.is_available(), "GPU allocation required"'
mkdir -p "$RUN_DIR"
cp "$0" "$RUN_DIR/launch.sh"
python3 -u src/train_model.py \
  --model EleutherAI/pythia-2.8b --length 64 --max_length 128 \
  --epochs 4 --lr 2e-5 --kl_weight 0.7 --budget_mult 1.5 \
  --save_path "$RUN_DIR/stude" 2>&1 | tee "$RUN_DIR/stude.log"
python3 -u src/train_comparison_baselines.py \
  --model EleutherAI/pythia-2.8b --length 64 --max_length 128 \
  --methods del spe --budget_alpha 0.005 --epochs 3 --del_lr 5e-5 \
  --sparsity 0.9 --spe_lr 1e-7 --spe_damping 0.01 --spe_max_update 1.0 \
  --fp32 --seed 42 --output "$RUN_DIR/baselines" \
  2>&1 | tee "$RUN_DIR/del-spe.log"
python3 -u src/evaluate_saved_lm.py \
  --model EleutherAI/pythia-2.8b --run-dir "$RUN_DIR" \
  --output "$RUN_DIR/lm-eval-full" \
  2>&1 | tee "$RUN_DIR/lm-eval-full.log"
