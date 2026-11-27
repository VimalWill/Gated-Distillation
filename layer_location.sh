#!/usr/bin/env bash
# Rebuttal: STUDE's selected band vs early / late / random contiguous / random
# scattered layers, matched budgets. See src/layer_location_experiment.py.
#
# Run inside an HPC GPU allocation from an activated environment:
#   bash layer_location.sh OUT_DIR prep          # split + dev band selection + plan
#   bash layer_location.sh OUT_DIR train INDEX   # one planned run (job-array task)
#   bash layer_location.sh OUT_DIR train-all     # every planned run, sequentially
#   bash layer_location.sh OUT_DIR summarize     # dev-selected test results table
#
# Example job array after prep (N = number of runs printed by prep):
#   sbatch --array=0-$((N-1)) --gpus=1 --wrap "bash layer_location.sh $OUT train \$SLURM_ARRAY_TASK_ID"
# Train runs are resumable: a run with an existing result.json is skipped.
set -euo pipefail
cd "$(dirname "$0")"
OUT="${1:?Usage: bash layer_location.sh OUT_DIR prep|train INDEX|train-all|summarize}"
STAGE="${2:?missing stage}"
MODEL="${MODEL:-EleutherAI/pythia-2.8b}"
PY="python3 -u src/layer_location_experiment.py"
mkdir -p "$OUT/logs"

case "$STAGE" in
  prep)
    $PY split --length 64 --split-seed 0 --dev-fraction 0.5 --out "$OUT/splits.json"
    $PY select-band --model "$MODEL" --splits "$OUT/splits.json" --width 15 \
        --prune-ratio 0.10 --out "$OUT/band.json" 2>&1 | tee "$OUT/logs/select-band.log"
    $PY plan --band "$OUT/band.json" --splits "$OUT/splits.json" --paper-band 10 24 \
        --random-draws 3 --seeds 0 1 2 --lrs 1e-5 2e-5 4e-5 --kl-weights 0.7 \
        --epochs 4 --grad-accum 32 --budget-mult 1.5 --out "$OUT/plan.json" \
        | tee "$OUT/logs/plan.log"
    ;;
  train)
    IDX="${3:?missing run index}"
    $PY train --plan "$OUT/plan.json" --index "$IDX" --out "$OUT" 2>&1 | tee "$OUT/logs/train-$IDX.log"
    ;;
  train-all)
    N=$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))['runs']))" "$OUT/plan.json")
    for ((i = 0; i < N; i++)); do
      $PY train --plan "$OUT/plan.json" --index "$i" --out "$OUT" 2>&1 | tee "$OUT/logs/train-$i.log"
    done
    ;;
  summarize)
    $PY summarize --plan "$OUT/plan.json" --out "$OUT"
    ;;
  *) echo "Unknown stage: $STAGE" >&2; exit 1 ;;
esac
