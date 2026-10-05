"""Offline regression checks for controlled initialization and training."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiment_protocol import make_splits, read_json, write_json
from research_experiments import controlled, supervised_training


class ControlledTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def test_float32_overrides_hub_precision_and_preserves_exposures(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import (AutoModelForCausalLM, GPT2Config, GPTNeoXConfig,
                                  LlamaConfig, PreTrainedTokenizerFast)

        # These sources declare low precision, as real Hub configs do. The
        # first Pythia-shaped case produced a nonfinite loss before the fix.
        architectures = [
            GPTNeoXConfig(vocab_size=16, hidden_size=16, intermediate_size=32,
                          num_hidden_layers=1, num_attention_heads=2,
                          max_position_embeddings=16, torch_dtype="float16"),
            LlamaConfig(vocab_size=16, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                        max_position_embeddings=16, torch_dtype="bfloat16"),
            GPT2Config(vocab_size=16, n_embd=16, n_layer=1, n_head=2,
                       n_positions=16, torch_dtype="float16"),
        ]
        backend = Tokenizer(WordLevel({"[PAD]": 0, "[EOS]": 1, "[UNK]": 2,
                                      **{f"word{i}": i + 3 for i in range(13)}},
                                     unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend,
                                            pad_token="[PAD]", eos_token="[EOS]",
                                            unk_token="[UNK]")
        rows = [{"input": f"word{i} word{i+1} word{i+2}", "label": i % 2}
                for i in range(8)]
        bundle = make_splits(rows)
        member_ids = {r["id"] for group in bundle["roles"].values()
                      for r in group if r["label"] == 1}

        for architecture in architectures:
            with self.subTest(architecture=architecture.model_type), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                architecture.save_pretrained(root / "source")
                tokenizer.save_pretrained(root / "source")
                write_json(root / "splits.json", bundle)
                write_json(root / "config.json", {
                    "model": str(root / "source"), "splits": str(root / "splits.json"),
                    "output": str(root / "out"), "dtype": "float32", "device": "cpu",
                    "seeds": [42], "duplication_levels": [1, 2], "epochs": 2, "lr": 5e-4,
                })
                controlled(SimpleNamespace(config=str(root / "config.json")))
                for copies in (1, 2):
                    checkpoint = root / "out" / f"copies-{copies}" / "seed-42"
                    manifest = read_json(checkpoint / "membership_manifest.json")
                    training = manifest["training"]
                    self.assertEqual(manifest["dtype"], "torch.float32")
                    self.assertEqual(manifest["membership_basis"], "known_from_scratch")
                    self.assertEqual(set(training["exposures"]), member_ids)
                    self.assertEqual(training["optimizer_steps"], len(member_ids) * copies * 2)
                    for exposure in training["exposures"].values():
                        self.assertEqual(exposure["presentations"], copies * 2)
                    restored = AutoModelForCausalLM.from_pretrained(checkpoint)
                    self.assertTrue(all(torch.isfinite(p).all() for p in restored.parameters()))

    def test_nonfinite_gradient_stops_before_parameter_update(self):
        import torch

        class NonfiniteBackward(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                return value.clone()

            @staticmethod
            def backward(ctx, gradient):
                return torch.full_like(gradient, float("nan"))

        class Dummy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(()))

            def forward(self, **kwargs):
                return SimpleNamespace(loss=NonfiniteBackward.apply(self.weight))

        model = Dummy()
        rows = [{"id": "sample", "input": "two tokens"}]
        tokenizer = lambda text, **kwargs: {"input_ids": [0, 1]}
        with self.assertRaisesRegex(FloatingPointError, "Gradient clipping failed.*optimizer_steps=0"):
            supervised_training(model, tokenizer, rows, {"seed": 42})
        self.assertEqual(model.weight.item(), 1.)


if __name__ == "__main__":
    unittest.main()
