"""Create and run a tiny offline transformer experiment (not scientific evidence)."""
import argparse
import json
from pathlib import Path

from experiment_protocol import make_splits, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/smoke")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    if root.exists():
        raise FileExistsError(root)
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, PreTrainedTokenizerFast
    from research_experiments import run
    torch.set_num_threads(1)
    torch.manual_seed(42)
    vocabulary = {"[UNK]": 0, "[EOS]": 1, "[PAD]": 2}
    vocabulary.update({f"word{i}": i + 3 for i in range(61)})
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                        eos_token="[EOS]", pad_token="[PAD]")
    model = GPTNeoXForCausalLM(GPTNeoXConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                                            num_hidden_layers=2, num_attention_heads=2,
                                            max_position_embeddings=32, bos_token_id=1,
                                            eos_token_id=1, pad_token_id=2))
    model.save_pretrained(root / "model")
    tokenizer.save_pretrained(root / "model")
    rows = [{"input": " ".join(f"word{(i + j) % 61}" for j in range(10)), "label": i % 2}
            for i in range(32)]
    write_json(root / "splits.json", make_splits(rows))
    utility = [{"text": f"word{i} word60 word{i} word59 word{i} word58"} for i in range(30)]
    (root / "utility.jsonl").write_text("".join(json.dumps(r) + "\n" for r in utility))
    config = {"model": str(root / "model"), "output": str(root / "runs"),
              "splits": str(root / "splits.json"), "utility_jsonl": str(root / "utility.jsonl"),
              "device": "cpu", "dtype": "float32", "seeds": [42],
              "selectors": ["middle"], "variants": ["STUDE"], "baselines": [],
              "layer_counts": [1], "epochs": 1, "tuning_grid": {"lr": [0.0001], "kl_weight": [0.1]},
              "max_length": 16, "prefix_length": 3, "continuation_length": 3,
              "utility_tokens": 64, "gradient_accumulation": 2,
              "utility_budget_mult": 10, "bootstrap": 20, "plots": False}
    write_json(root / "config.json", config)
    run(argparse.Namespace(config=str(root / "config.json")))
    print(f"Offline smoke experiment completed: {root / 'runs' / 'aggregate.json'}")


if __name__ == "__main__":
    main()
