# Paper experiments

## Train saved models and run full LM-Eval

The active workflow trains STUDE and applies DEL/SPE to pretrained
Pythia-2.8B on WikiMIA length64, saves each edited checkpoint, then evaluates
Baseline, DEL, SPE, and STUDE on all six downstream task splits.
Run from the repository in an activated HPC environment with a GPU allocation:

```bash
RUN_DIR="/work/nvme/bgly/$USER/gated-distillation/pythia-full-lm-$(date +%Y%m%d-%H%M%S)"
srun bash train_pythia_full_lm.sh "$RUN_DIR"
```

The launcher uses the paper's documented Pythia hyperparameters. It does not
establish the tuned settings for other backbones. It saves STUDE under `stude/`,
DEL/SPE under `baselines/`, and full LM-Eval scores, configuration, and per-method
details under `lm-eval-full/`. Each stage has a separate log. A new run directory
is required; stage failures stop the pipeline.

To repeat only evaluation using saved weights, choose a fresh output directory:

```bash
srun python3 -u src/evaluate_saved_lm.py \
  --model EleutherAI/pythia-2.8b \
  --run-dir "$RUN_DIR" \
  --output "$RUN_DIR/lm-eval-repeat"
```

This reproduces the existing comparison's data roles: WikiMIA members are used
for forgetting and nonmembers for retention. The membership panel is reused for
updates/selection and evaluation; this is not an isolated held-out protocol.
Full downstream task coverage does not change that limitation.

## Full evaluation with the legacy comparison commands

`compare_unlearning.py --lm_eval` and `memorization_effect.py --lm_eval` now
evaluate each requested lm-eval task's full evaluation split by default, replacing
the previous 200-example default. `--lm_eval_limit 0` explicitly requests the same
full evaluation. MMLU remains 5-shot and the other default tasks remain zero-shot.

The membership table (ROC-AUC, Min-K, PPL-AUC, Lowercase, Zlib, and Smaller-Ref)
already uses the complete selected WikiMIA split. Its separate utility PPL probe
in `compare_unlearning.py` now uses all WikiText-2 test lines satisfying its
existing minimum-length filter, replacing the previous 64-line default. The
utility probe in `sweep_unlearning.py` uses the same full-corpus default.
`--n_utility 0` explicitly requests all eligible lines. Token context lengths,
method hyperparameters, and the legacy PPL averaging definition are unchanged.
These full-corpus values must be distinguished from earlier capped results.

For example, retaining an existing checkpoint and frozen DEL/SPE settings:

```bash
python3 src/compare_unlearning.py \
  --model EleutherAI/pythia-2.8b --length 64 \
  --checkpoint /path/to/existing/stude-checkpoint \
  --methods del spe --budget_alpha 0.005 --epochs 3 --del_lr 5e-5 --spe_lr 1e-7 \
  --fp32 --lm_eval --lm_eval_limit 0 --n_utility 0
```

Supply each backbone's existing checkpoint and original method settings to rerun
the corresponding comparison. This legacy command still reuses its WikiMIA panel;
full evaluation does not turn it into the isolated research protocol.

## Shared research utilities

`research_experiments.py`, `experiment_protocol.py`, `research_metrics.py`, and
`layer_selection.py` retain the general split, metric, and intervention tools.
The explicit-config entrypoint remains `./run.sh /path/to/config.json`.
Their presence is not evidence of completed or validated paper experiments.
Tests live under `tests/` and can be run with:

```bash
python3 -m unittest discover -s tests -v
```

The discarded scratch-training pilot launchers and local generated artifacts
have been removed. Raw downloaded records remain outside the repository.

## Layer weight-rank profiles

Measure per-layer spectral-entropy effective rank directly from pretrained
checkpoint weights. This needs no WikiMIA or other text data, but computing SVDs
for all layer matrices takes GPU time:

```bash
srun python3 -u src/plot_layer_weight_ranks.py \
  --output /work/nvme/bgly/$USER/gated-distillation/layer-weight-ranks-$(date +%Y%m%d-%H%M%S)
```

The output contains a combined CSV, JSON metadata, and PDF plot. It pools squared
singular values from each block's Linear weight matrices and also reports a
normalized curve. This block-level statistic is distinct from both activation
rank and an individual-matrix rank. Compare its shape with the layerwise pruning
memorization results; similarity is a hypothesis to test, not an assumption.
