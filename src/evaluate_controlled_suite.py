"""Evaluate every saved checkpoint from run_controlled_gh200.py."""
import argparse
import sys
from pathlib import Path

from experiment_protocol import read_json, validate_splits, write_json


MODELS = ("pythia-160m", "Llama-3.2-1B", "gpt2")
COPIES = (1, 5, 10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True,
                        help="Controlled run directory containing splits.json and model folders")
    parser.add_argument("--output", help="Fresh evaluation output directory; default RUN_DIR/evaluation-full")
    parser.add_argument("--lm-eval", action="store_true",
                        help="Also run full lm-eval test splits for every checkpoint")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    split_path = run_dir / "splits.json"
    if not split_path.is_file():
        parser.error(f"Missing controlled split file: {split_path}")
    bundle = validate_splits(read_json(split_path))

    evaluation_root = Path(args.output).resolve() if args.output else run_dir / "evaluation-full"
    if evaluation_root.exists():
        parser.error(f"Evaluation output already exists; choose a fresh --output: {evaluation_root}")

    jobs = []
    for model_name in MODELS:
        for copies in COPIES:
            checkpoint = run_dir / model_name / "checkpoints" / f"copies-{copies}" / "seed-42"
            manifest_path = checkpoint / "membership_manifest.json"
            if not manifest_path.is_file():
                parser.error(f"Missing completed training manifest: {manifest_path}")
            membership = read_json(manifest_path)
            if membership.get("dataset_hash") != bundle["dataset_hash"]:
                parser.error(f"Checkpoint/split membership mismatch: {checkpoint}")
            result_dir = evaluation_root / "runs" / model_name / f"copies-{copies}"
            if result_dir.exists():
                parser.error(f"Evaluation already exists: {result_dir}")
            jobs.append((model_name, copies, checkpoint, membership, result_dir))

    if args.lm_eval:
        try:
            import lm_eval  # noqa: F401
        except ImportError:
            parser.error("lm-eval is required for --lm-eval; install it in the active HPC environment")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from research_experiments import run
    from lm_eval_utils import DEFAULT_TASKS

    evaluation_root.mkdir(parents=True)
    config_dir = evaluation_root / "configs"
    config_dir.mkdir()
    for model_name, copies, checkpoint, membership, result_dir in jobs:
        config_path = config_dir / f"{model_name}-copies-{copies}.json"
        config = dict(
            membership["config"],
            model=str(checkpoint),
            output=str(result_dir),
            splits=str(split_path),
            backbone_identifier=membership["config"].get("backbone_identifier", model_name),
            initialization="scratch",
            device="cuda",
            dtype="float32",
            seeds=[42],
            selectors=[],
            variants=[],
            baselines=[],
            max_length=512,
            prefix_length=32,
            continuation_length=32,
            utility_tokens=None,
            bootstrap=1000,
            plots=False,
            downstream=args.lm_eval,
            downstream_tasks=list(DEFAULT_TASKS),
            downstream_limit=None,
            downstream_batch_size=4,
            purpose="Full-split controlled membership, extraction, WikiText utility, and optional lm-eval",
        )
        write_json(config_path, config)
        print(f"Evaluating {model_name}, copies={copies}: {checkpoint}", flush=True)
        run(argparse.Namespace(config=str(config_path)))

    print(f"All nine checkpoints evaluated. Results: {evaluation_root}", flush=True)


if __name__ == "__main__":
    main()
