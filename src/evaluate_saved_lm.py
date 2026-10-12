"""Run full LM-Eval splits on baseline and saved DEL, SPE, and STUDE models."""
import argparse
import json
from pathlib import Path

from lm_eval_utils import DEFAULT_TASKS, print_lm_eval_table, run_lm_eval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True, help="Fresh results directory")
    parser.add_argument("--stude-checkpoint", help="STUDE path; defaults to RUN_DIR/stude")
    parser.add_argument("--del-checkpoint", help="DEL path; defaults to RUN_DIR/baselines/del")
    parser.add_argument("--spe-checkpoint", help="SPE path; defaults to RUN_DIR/baselines/spe")
    parser.add_argument("--methods", nargs="+", choices=("del", "spe", "stude"),
                        default=("del", "spe", "stude"),
                        help="Edited checkpoints to score, alongside Baseline")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    import torch
    import lm_eval  # Require the harness before starting any evaluation.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root = Path(args.run_dir)
    checkpoint_paths = {"del": args.del_checkpoint or str(root / "baselines/del"),
                        "spe": args.spe_checkpoint or str(root / "baselines/spe"),
                        "stude": args.stude_checkpoint or str(root / "stude")}
    sources = {"Baseline": args.model}
    sources.update({method.upper(): checkpoint_paths[method] for method in args.methods})
    for name, source in sources.items():
        if name != "Baseline" and not (Path(source) / "config.json").is_file():
            parser.error(f"Missing {name} checkpoint: {source}")
    if not torch.cuda.is_available():
        parser.error("A GPU allocation is required")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(
        dict(sources=sources, tasks=DEFAULT_TASKS, limit=None, seed=42,
             dtype="float32", batch_size=args.batch_size), indent=2) + "\n")
    # All edited checkpoints retain the base model's vocabulary. Load the
    # tokenizer from the base model once; checkpoint folders may contain only
    # weights/config and may not have a complete standalone tokenizer bundle.
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    scores = {}
    for name, source in sources.items():
        model = AutoModelForCausalLM.from_pretrained(
            source, torch_dtype=torch.float32).to("cuda").eval()
        scores[name] = run_lm_eval(
            model, tokenizer, tasks=DEFAULT_TASKS, limit=None,
            batch_size=args.batch_size, seed=42,
            details_path=output / name / "downstream_details.json")
        (output / "scores.json").write_text(json.dumps(scores, indent=2) + "\n")
        if set(scores[name]) != set(DEFAULT_TASKS):
            raise RuntimeError(f"Incomplete LM-Eval for {name}; inspect its details file")
        del model
        torch.cuda.empty_cache()
    print_lm_eval_table(scores)


if __name__ == "__main__":
    main()
