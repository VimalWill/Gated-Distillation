# Experiment protocol and first pilot

The first validated path is baseline versus one middle-layer STUDE update. It
uses disjoint selection, tuning, optimization, and evaluation records. The
broader selectors, controlled-membership, LoRA, DEL/SPE, and durability paths
are experimental until separately exercised; their presence does not establish
that the corresponding research experiments have been completed.

## Run the small pilot

```bash
python3 -m unittest discover -s tests -v
python3 src/smoke_experiment.py --output artifacts/smoke-001
python3 src/prepare_pilot.py --output artifacts/pilot-001
./run.sh artifacts/pilot-001/config.json
```

The smoke command creates a tiny random Pythia-style model and synthetic data
offline. It checks execution only. Choose a new output directory for each run;
the runner refuses to overwrite existing experiments.

The pilot preparation command expects a locally cached Pythia-160M checkpoint
and WikiMIA length128 Arrow dataset. It downloads WikiText-2 unless an existing
`--utility-jsonl` is supplied. Configuration records the exact local model
snapshot. The pilot runs on CPU with four threads, 32 examples per membership
class, one seed, one epoch, and two middle layers. It prunes 10% of attention
weights across the model before optimizing the selected blocks. This measures
the combined pruning/unlearning intervention; it does not isolate the two effects.

`./run.sh` without a configuration retains the older exploratory workflow, now
including the trained checkpoint in comparison. That legacy workflow reuses its
WikiMIA panel and must not be used for final research comparisons.

## What is isolated and saved

- Content-normalized IDs and checks reject duplicate content across roles.
  Split assignments and a dataset fingerprint are saved. Near-duplicate or
  paraphrase detection is not implemented.
- Optimization uses only member examples in the optimization role. Epoch and
  hyperparameter selection uses only the tuning role. The final evaluation role
  does not drive checkpoint selection. Optimization examples are reported
  separately as an in-sample diagnostic.
- Utility documents are split into selection, tuning, and evaluation corpora,
  excluding exact normalized experimental examples. The tokenized corpora,
  document IDs, token hashes, and actual token counts are saved.
- Perplexity is `exp(total NLL / prediction tokens)`, including partial final
  blocks. Nonfinite loss fails the run; it is not dropped from averages.
- The default pilot uses 1,024 utility tokens per role and a 64-token context.
  These deliberately small limits are for execution validation, not final
  utility claims. Final runs should use a substantially larger token budget.
- Attack direction is chosen on tuning scores and frozen for evaluation.
  Reports include raw AUC, oriented AUC, descriptive `max(AUC, 1-AUC)`,
  stratified-bootstrap intervals, and `max_threshold(TPR-FPR)` advantage.
  Low-FPR TPR includes the allowed false-positive count; small pilots cannot
  resolve low-FPR behavior reliably.
- Extraction uses greedy generation with explicit prefix/continuation lengths.
  Exact match compares the complete token continuation. Token recovery is
  positional; longest matching span is a contiguous common substring allowing
  different offsets. Short examples are counted as skipped.
- Manifests record configuration, code revision/dirty status, packages, selected
  layers, trainable and pruned tensors, actual update counts, deployment zeros,
  and checkpoint paths. A dirty checkout is recorded but is not a source snapshot;
  preserve the code/commit with archived scientific runs.

Artifacts are under `<output>/seed-<seed>/`: `manifest.json`,
`utility_corpora.json`, `baseline/`, each method's `tuning.json`, `manifest.json`,
per-example `scores.json`, `metrics.json`, and `results.json`. Across-seed means,
standard deviations, and raw values are saved in `aggregate.json`.

WikiMIA membership labels remain unverified for the chosen backbone. A successful
pilot is not proof of known-membership unlearning. The `controlled` command can
construct exposure ledgers; pretrained initialization establishes only added
exposure, while scratch initialization can establish complete membership.

## Next runs after the pilot

1. Increase evaluation/utility counts and repeat across seeds.
2. Exercise DEL/SPE under this isolated protocol; give each a documented tuning
   grid. Their objectives differ from STUDE and are labeled as whole-method
   comparisons, not matched-selector ablations.
3. Exercise pruning and residual-masking loss-sensitivity profiles, independent
   layer selection, and measured fixed-width windows on each backbone.
4. Validate the LoRA merge path, controlled duplication, rank associations,
   and post-unlearning benign/relearning branches separately before large runs.

Residual masking is a FIT-style loss-sensitivity proxy, not a verified reproduction
of a named FIT algorithm. Causal tracing is not implemented. Activation effective
rank uses singular-value entropy of uncentered block outputs; its associations
are descriptive and do not establish a connection to a theorem by themselves.

The optional `neighbors` command builds lexical Jaccard neighbors. Semantic
relevance requires curator review. Optional downstream evaluation preserves
harness counts and uncertainty in a details file; unavailable/failed evaluations
must not be interpreted as completed benchmarks.

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

## Evaluate controlled-exposure checkpoints

After all nine model/copy checkpoints have `membership_manifest.json`, run the
suite on a GPU:

```bash
srun python3 -u src/evaluate_controlled_suite.py \
  --run-dir /work/nvme/bgly/$USER/gated-distillation/controlled-3313516 \
  --lm-eval
```

The evaluator checks the saved split fingerprint for every checkpoint and saves
per-example scores, membership and extraction metrics, token-weighted utility
loss on every eligible WikiText-2 test line, and full lm-eval results to
`RUN_DIR/evaluation-full/`. Install `lm-eval` in the active environment before
using `--lm-eval`. Each evaluation output is immutable; choose a fresh `--output`
directory to repeat it. The membership evaluation uses the saved controlled
panel (64 examples per class), not all 542 records in WikiMIA length64.
