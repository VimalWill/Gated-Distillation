"""Offline checks for split leakage, metric semantics, and layer interventions."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiment_protocol import make_splits, validate_splits
from research_metrics import attack_metrics, auc, extraction_metrics, longest_matching_span


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"input": f"example {i} label {label}", "label": label}
                     for label in (0, 1) for i in range(12)]

    def test_splits_are_disjoint_and_order_independent(self):
        first = make_splits(self.rows, 9)
        second = make_splits(self.rows[::-1], 9)
        self.assertEqual(first, second)
        self.assertEqual(validate_splits(first), first)
        ids = [r["id"] for rows in first["roles"].values() for r in rows]
        self.assertEqual(len(ids), len(set(ids)))

    def test_leak_and_conflicting_membership_rejected(self):
        bundle = make_splits(self.rows)
        bundle["roles"]["evaluation"].append(bundle["roles"]["optimization"][0])
        with self.assertRaises(ValueError):
            validate_splits(bundle)
        with self.assertRaises(ValueError):
            make_splits(self.rows + [{"input": self.rows[0]["input"], "label": 1}])

    def test_tied_and_reversed_auc(self):
        self.assertEqual(auc([0, 1], [1, 1]), 0.5)
        metrics = attack_metrics([0, 0, 1, 1], [4, 3, 2, 1], direction=-1, bootstrap=20)
        self.assertEqual(metrics["raw_auc"], 0)
        self.assertEqual(metrics["oriented_auc"], 1)
        self.assertEqual(metrics["max_auc"], 1)
        self.assertEqual(metrics["advantage"], 1)
        self.assertEqual(metrics["ci95"]["oriented_auc"], [1, 1])

    def test_extraction_has_positional_recovery_and_contiguous_span(self):
        result = extraction_metrics([1, 2, 3, 4], [0, 1, 2, 3])
        self.assertEqual(result["exact_match"], 0)
        self.assertEqual(result["token_recovery"], 0)
        self.assertEqual(result["longest_matching_span"], 3)
        self.assertEqual(longest_matching_span([1, 2, 3], [1, 9, 3]), 1)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def test_regions_restore_and_only_prune_requested_layers(self):
        import torch
        from transformers import (GPTNeoXConfig, GPTNeoXForCausalLM, LlamaConfig,
                                  LlamaForCausalLM, GPTNeoConfig, GPTNeoForCausalLM)
        from layer_selection import attention_parameters, intervention, layer_index
        models = [
            GPTNeoXForCausalLM(GPTNeoXConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                            num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=32)),
            LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2)),
            GPTNeoForCausalLM(GPTNeoConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                          num_layers=2, num_heads=2, attention_types=[[["global"], 2]])),
        ]
        for model in models:
            before = {n: p.detach().clone() for n, p in attention_parameters(model).items()}
            with self.assertRaisesRegex(RuntimeError, "restore"):
                with intervention(model, [1], ratio=0.5):
                    for name, parameter in attention_parameters(model).items():
                        self.assertEqual(torch.equal(before[name], parameter), layer_index(name) == 0)
                    raise RuntimeError("restore")
            for name, parameter in attention_parameters(model).items():
                self.assertTrue(torch.equal(before[name], parameter))

    def test_corpus_loss_weights_prediction_tokens(self):
        import torch
        from types import SimpleNamespace
        from research_metrics import corpus_loss
        class Dummy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.config = SimpleNamespace(max_position_embeddings=16)
            def forward(self, input_ids, labels):
                return SimpleNamespace(loss=torch.tensor(float(input_ids[0, 0])))
        # Two predicted tokens at NLL=1; four at NLL=3 -> 14/6, not 2.
        result = corpus_loss(Dummy(), {"blocks": [[1, 0, 0], [3, 0, 0, 0, 0]], "token_hash": "test"})
        self.assertAlmostEqual(result["nll"], 14 / 6)
        self.assertEqual(result["n_tokens"], 6)


if __name__ == "__main__":
    unittest.main()
