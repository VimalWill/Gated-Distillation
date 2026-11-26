#!/usr/bin/env bash
# Run inside an active HPC GPU allocation, from an activated Python environment.
#
# Usage: bash train_pythia_full_lm.sh RUN_DIRECTORY [--stude] [--del] [--spe] [--eval]
#   No stage flags = run every stage in order (STUDE, DEL, SPE, eval).
#   Stages may be run as separate jobs against the same RUN_DIRECTORY; each
#   stage refuses to overwrite its own existing output.
#   --eval scores the Baseline plus whichever of STUDE/DEL/SPE checkpoints
#   exist in RUN_DIRECTORY, on the full LM-Eval splits (no sample cap).
set -euo pipefail
cd "$(dirname "$0")"

usage() { echo "Usage: bash $0 RUN_DIRECTORY [--stude] [--del] [--spe] [--eval]" >&2; exit 1; }
[[ $# -ge 1 && "$1" != --* ]] || usage
RUN_DIR="$1"; shift

RUN_STUDE=0 RUN_DEL=0 RUN_SPE=0 RUN_EVAL=0
if [[ $# -eq 0 ]]; then
  RUN_STUDE=1 RUN_DEL=1 RUN_SPE=1 RUN_EVAL=1
fi
for arg in "$@"; do
  case "$arg" in
    --stude) RUN_STUDE=1 ;;
    --del)   RUN_DEL=1 ;;
    --spe)   RUN_SPE=1 ;;
    --eval)  RUN_EVAL=1 ;;
    *) echo "Unknown option: $arg" >&2; usage ;;
  esac
done

MODEL="EleutherAI/pythia-2.8b"
STAMP="$(date +%Y%m%d-%H%M%S)"

python3 -c 'import torch, transformers, datasets, sklearn, lm_eval; assert torch.cuda.is_available(), "GPU allocation required"'
mkdir -p "$RUN_DIR"
cp "$0" "$RUN_DIR/launch-$STAMP.sh"

if [[ $RUN_STUDE -eq 1 ]]; then
  if [[ -e "$RUN_DIR/stude" ]]; then
    echo "STUDE output already exists: $RUN_DIR/stude" >&2; exit 1
  fi
  python3 -u src/train_model.py \
    --model "$MODEL" --length 64 --max_length 128 \
    --epochs 4 --lr 2e-5 --kl_weight 0.7 --budget_mult 1.5 \
    --save_path "$RUN_DIR/stude" 2>&1 | tee "$RUN_DIR/stude-$STAMP.log"
fi

run_baseline() {  # $1 = del | spe
  python3 -u src/train_comparison_baselines.py \
    --model "$MODEL" --length 64 --max_length 128 \
    --methods "$1" --budget_alpha 0.005 --epochs 3 --del_lr 5e-5 \
    --sparsity 0.9 --spe_lr 1e-7 --spe_damping 0.01 --spe_max_update 1.0 \
    --fp32 --seed 42 --output "$RUN_DIR/baselines" \
    2>&1 | tee "$RUN_DIR/$1-$STAMP.log"
}
if [[ $RUN_DEL -eq 1 ]]; then run_baseline del; fi
if [[ $RUN_SPE -eq 1 ]]; then run_baseline spe; fi

if [[ $RUN_EVAL -eq 1 ]]; then
  METHODS=()
  [[ -f "$RUN_DIR/stude/config.json" ]]         && METHODS+=(stude)
  [[ -f "$RUN_DIR/baselines/del/config.json" ]] && METHODS+=(del)
  [[ -f "$RUN_DIR/baselines/spe/config.json" ]] && METHODS+=(spe)
  echo "Evaluating Baseline + ${METHODS[*]:-(no edited checkpoints found)} on full LM-Eval splits"
  python3 -u src/evaluate_saved_lm.py \
    --model "$MODEL" --run-dir "$RUN_DIR" \
    --output "$RUN_DIR/lm-eval-full-$STAMP" --methods ${METHODS[@]+"${METHODS[@]}"} \
    2>&1 | tee "$RUN_DIR/lm-eval-full-$STAMP.log"
fi
