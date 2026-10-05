"""Run the first controlled-membership pilot across three model families."""
import argparse
import os
from pathlib import Path
import random
from types import SimpleNamespace

from experiment_protocol import canonical_records, make_splits, write_json
from research_experiments import controlled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/controlled-gh200-001")
    parser.add_argument("--models", nargs="+", default=[
        "EleutherAI/pythia-160m", "meta-llama/Llama-3.2-1B", "openai-community/gpt2"])
    parser.add_argument("--per-class", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--duplication-levels", nargs="+", type=int, default=[1, 5, 10])
    args = parser.parse_args()
    if args.per_class < 4 or args.epochs < 1:
        parser.error("Use at least four examples per class and positive epochs")
    if any(c < 1 for c in args.duplication_levels):
        parser.error("Duplication levels must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be distinct")
    names = [model.rstrip("/").split("/")[-1] for model in args.models]
    if len(set(names)) != len(names):
        parser.error("Model names must be distinct")

    import torch
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: check the Slurm GPU allocation")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    root = Path(args.output).resolve()
    if root.exists():
        raise FileExistsError(f"Choose a fresh --output directory: {root}")

    # Verify access to every model before training. Scratch needs no weights.
    snapshots = [snapshot_download(model, allow_patterns=[
        "*.json", "*.model", "*.txt", "*.tiktoken"])
        for model in args.models]
    rows = canonical_records(load_dataset("swj0419/WikiMIA", split="WikiMIA_length128"))
    selected = []
    for label in (0, 1):
        candidates = [r for r in rows if r["label"] == label]
        if len(candidates) < args.per_class:
            raise ValueError(f"Insufficient examples for class {label}")
        selected.extend(random.Random(42).sample(candidates, args.per_class))
    root.mkdir(parents=True, exist_ok=False)
    splits = root / "splits.json"
    write_json(splits, make_splits(selected, seed=42))

    for name, model_id, snapshot in zip(names, args.models, snapshots):
        config_path = root / f"{name}.json"
        write_json(config_path, {
            "model": snapshot, "backbone_identifier": model_id,
            "revision": Path(snapshot).name,
            "splits": str(splits), "output": str(root / name / "checkpoints"),
            "initialization": "scratch", "device": "cuda", "dtype": "float32",
            "seeds": args.seeds, "duplication_levels": args.duplication_levels,
            "epochs": args.epochs, "lr": 5e-4, "max_length": 128,
            "purpose": "Controlled-exposure pilot; duplication also changes update count",
        })
        print(f"Starting {model_id}", flush=True)
        controlled(SimpleNamespace(config=str(config_path)))


if __name__ == "__main__":
    main()
