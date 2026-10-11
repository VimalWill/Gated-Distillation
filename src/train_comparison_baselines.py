"""Apply DEL/SPE to pretrained weights and save reusable checkpoints.

Uses the legacy Table 1 protocol: WikiMIA members are the forget set and
nonmembers are the retain set. This does not create held-out evaluation splits.
"""
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed

from compare_unlearning import (
    METHODS, attention_weight_names, build_parser, load_model, make_loader,
)


def main():
    parser = build_parser()
    parser.description = __doc__
    parser.add_argument("--output", required=True, help="New checkpoint parent directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if any(method not in ("del", "spe") for method in args.methods):
        parser.error("Only --methods del spe are supported")
    if args.checkpoint or args.lm_eval:
        parser.error("This command saves models only; omit --checkpoint and --lm_eval")
    output = Path(args.output)
    if output.exists():
        parser.error(f"Output already exists; choose a fresh directory: {output}")
    if not torch.cuda.is_available():
        parser.error("Run this command inside an HPC GPU allocation")

    device = torch.device("cuda")
    dtype = torch.float32 if args.fp32 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{args.length}")
    forget = [row["input"] for row in dataset if row["label"] == 1]
    retain = [row["input"] for row in dataset if row["label"] == 0]
    output.mkdir(parents=True, exist_ok=False)
    for method in args.methods:
        set_seed(args.seed)
        print(f"Applying {method} to {args.model}, WikiMIA length {args.length}", flush=True)
        model = load_model(args.model, device, dtype)
        forget_loader = make_loader(forget, tokenizer, args.batch_size, args.max_length)
        retain_loader = make_loader(retain, tokenizer, args.batch_size, args.max_length)
        METHODS[method](model, tokenizer, forget_loader, retain_loader, retain[:32],
                        device, attention_weight_names(model), args)
        destination = output / method
        model.save_pretrained(destination)
        tokenizer.save_pretrained(destination)
        manifest = dict(vars(args), method=method, status="complete",
                        forget_count=len(forget), retain_count=len(retain),
                        dataset_fingerprint=getattr(dataset, "_fingerprint", None))
        (destination / "training_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n")
        print(f"Saved {destination}", flush=True)
        del model, forget_loader, retain_loader
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
