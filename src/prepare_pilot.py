"""Prepare a small Pythia CPU pilot from cached WikiMIA and held-out WikiText."""
import argparse
import json
from pathlib import Path
import random

from experiment_protocol import canonical_records, make_splits, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/pilot-001")
    parser.add_argument("--per-class", type=int, default=32)
    parser.add_argument("--utility-jsonl", help="Existing utility corpus; otherwise downloads WikiText-2")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    if (root / "config.json").exists():
        raise FileExistsError(root / "config.json")
    from datasets import Dataset, load_dataset
    cache = Path.home() / ".cache" / "huggingface"
    snapshots = sorted((cache / "hub/models--EleutherAI--pythia-160m/snapshots").glob("*"))
    snapshots = [p for p in snapshots if (p / "model.safetensors").exists()]
    sources = sorted((cache / "datasets/swj0419___wiki_mia").rglob("*WikiMIA_length128.arrow"))
    if len(snapshots) != 1 or len(sources) != 1:
        raise ValueError("Expected one cached Pythia-160M snapshot and WikiMIA length128 Arrow file")
    all_rows = canonical_records(Dataset.from_file(str(sources[0])))
    selected = []
    for label in (0, 1):
        rows = [r for r in all_rows if r["label"] == label]
        random.Random(42).shuffle(rows)
        if len(rows) < args.per_class:
            raise ValueError("Insufficient examples")
        selected.extend(rows[:args.per_class])
    splits = make_splits(selected, 42)
    splits["source"] = {"dataset": "swj0419/WikiMIA", "split": "WikiMIA_length128",
                        "arrow_file": str(sources[0]), "pilot_per_class": args.per_class}
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "splits.json", splits)
    if args.utility_jsonl:
        utility_path = Path(args.utility_jsonl).resolve()
    else:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", cache_dir=str(root / "cache"))
        utility_path = root / "utility.jsonl"
        utility_path.write_text("".join(json.dumps({"text": r["text"]}) + "\n" for r in dataset if r["text"].strip()))
    config = {"model": str(snapshots[0].resolve()), "revision": snapshots[0].name,
              "backbone_identifier": "EleutherAI/pythia-160m", "output": str(root / "runs"),
              "splits": str(root / "splits.json"), "utility_jsonl": str(utility_path),
              "device": "cpu", "dtype": "float32", "seeds": [42], "cpu_threads": 4,
              "selectors": ["middle"], "variants": ["STUDE"], "baselines": [],
              "layer_counts": [2], "epochs": 1, "tuning_grid": {"lr": [1e-5], "kl_weight": [0.1]},
              "prune_ratio": 0.1, "max_length": 64, "prefix_length": 16, "continuation_length": 16,
              "utility_tokens": 1024, "gradient_accumulation": 4, "utility_budget_mult": 1.5,
              "bootstrap": 200, "plots": False,
              "purpose": "CPU pilot: validates execution, not a paper-quality comparison"}
    write_json(root / "config.json", config)
    print(f"Prepared pilot: ./run.sh {root / 'config.json'}")


if __name__ == "__main__":
    main()
