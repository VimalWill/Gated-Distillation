"""
Optional downstream-capability evaluation via EleutherAI's lm-evaluation-harness.

Held-out perplexity is a proxy for "the model still works"; standard benchmark
accuracy is the real thing. A faithful unlearning method should barely move
these task accuracies, while a method that broke the model will tank them.

Kept optional (enabled by --lm_eval) because it is much slower/heavier than the
memorization panel. It wraps the already-loaded, possibly weight-mutated model
with lm-eval's HFLM, so it scores the exact post-unlearning model in memory —
not a checkpoint reloaded from disk.
"""

from typing import Dict, Iterable, Optional


# Reasoning/knowledge benchmarks. Protocol: MMLU 5-shot, the rest zero-shot.
DEFAULT_TASKS = ["piqa", "arc_easy", "arc_challenge", "hellaswag", "winogrande", "mmlu"]


def _fewshot_for(task: str) -> int:
    """Few-shot count per task: MMLU is 5-shot, everything else zero-shot."""
    return 5 if "mmlu" in task.lower() else 0


def run_lm_eval(
    model,
    tokenizer,
    tasks: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
    batch_size: int = 8,
) -> Dict[str, float]:
    """Return {task: accuracy} for `model` on the given lm-eval tasks.

    Uses acc_norm when a task reports it, else acc. `limit` caps examples per
    task — keep it small (e.g. 200), full suites are slow on multi-B models.
    Returns {} (with a printed note) if lm_eval isn't installed, so a missing
    dependency degrades gracefully instead of aborting the whole comparison.
    """
    task_list = list(tasks or DEFAULT_TASKS)
    try:
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        print("  [lm_eval not installed — skipping capability eval; `pip install lm-eval`]")
        return {}

    was_training = model.training
    model.eval()
    merged: Dict[str, dict] = {}
    try:
        lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
        # Group tasks by few-shot count and run each group separately, so MMLU
        # (5-shot) and the zero-shot tasks use their correct protocol in one pass.
        groups: Dict[int, list] = {}
        for t in task_list:
            groups.setdefault(_fewshot_for(t), []).append(t)
        for nshot, group_tasks in groups.items():
            out = simple_evaluate(model=lm, tasks=group_tasks,
                                  num_fewshot=nshot, limit=limit)
            if out and "results" in out:
                merged.update(out["results"])
    except Exception as e:
        # Optional add-on: never let an lm-eval API/version hiccup abort the
        # primary memorization comparison — degrade to an empty result.
        print(f"  [lm_eval failed — {type(e).__name__}: {e}; skipping capability eval]")
        if was_training:
            model.train()
        return {}
    if was_training:
        model.train()

    # Keep only the requested top-level tasks/groups (drops MMLU's 57 subtasks,
    # keeping just the aggregated `mmlu` score).
    scores: Dict[str, float] = {}
    for task in task_list:
        metrics = merged.get(task)
        if not metrics:
            continue
        acc = metrics.get("acc_norm,none", metrics.get("acc,none"))
        if acc is not None:
            scores[task] = float(acc)
    return scores


def print_lm_eval_table(results: Dict[str, Dict[str, float]]):
    """results: {model_name: {task: acc}} -> printed table with a Mean column."""
    results = {k: v for k, v in results.items() if v}
    if not results:
        return
    tasks = sorted({t for scores in results.values() for t in scores})
    if not tasks:
        return

    name_w = max(len(k) for k in results) + 2
    header = f"{'Model':<{name_w}}" + "".join(f"{t[:12]:>14}" for t in tasks) + f"{'Mean':>10}"
    print("\n" + "=" * len(header))
    print("LM-EVAL — downstream accuracy (higher = better; should stay near baseline)")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for name, scores in results.items():
        present = [scores[t] for t in tasks if t in scores]
        mean = sum(present) / len(present) if present else float("nan")
        row = f"{name:<{name_w}}"
        for t in tasks:
            row += f"{scores[t]:>14.4f}" if t in scores else f"{'N/A':>14}"
        row += f"{mean:>10.4f}"
        print(row)
    print("=" * len(header))
