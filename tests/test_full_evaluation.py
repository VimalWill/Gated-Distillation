"""Regression checks for uncapped comparison CLIs and harness calls."""
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import compare_unlearning
import memorization_effect
from experiment_protocol import read_json
from lm_eval_utils import run_lm_eval


class FullEvaluationTests(unittest.TestCase):
    def test_cli_defaults_and_explicit_full_override_evaluate_every_example(self):
        import torch

        for parser, flags in (
            (compare_unlearning.build_parser(), ["--lm_eval"]),
            (memorization_effect.build_parser(), ["--lm_eval"]),
            (compare_unlearning.build_parser(), ["--lm_eval", "--lm_eval_limit", "0"]),
            (memorization_effect.build_parser(), ["--lm_eval", "--lm_eval_limit", "none"]),
        ):
            args = parser.parse_args(flags)
            calls = []

            # Simulate tasks with more examples than the previous cap of 200.
            def simple_evaluate(model, tasks, num_fewshot, limit,
                                random_seed=None, numpy_random_seed=None,
                                torch_random_seed=None, fewshot_random_seed=None):
                total = 431 if tasks == ["piqa"] else 503
                evaluated = total if limit is None else min(total, limit)
                calls.append((tasks, num_fewshot, limit, evaluated))
                return {"results": {tasks[0]: {"acc,none": 0.5}},
                        "n-samples": {tasks[0]: {"original": total, "effective": evaluated}}}

            harness = types.ModuleType("lm_eval")
            harness.simple_evaluate = simple_evaluate
            models = types.ModuleType("lm_eval.models")
            huggingface = types.ModuleType("lm_eval.models.huggingface")
            huggingface.HFLM = lambda **kwargs: kwargs["pretrained"]
            modules = {"lm_eval": harness, "lm_eval.models": models,
                       "lm_eval.models.huggingface": huggingface}
            model = torch.nn.Linear(1, 1)
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "details.json"
                with patch.dict(sys.modules, modules):
                    scores = run_lm_eval(model, None, ["piqa", "mmlu"],
                                         limit=args.lm_eval_limit, details_path=path)
                self.assertEqual(scores, {"piqa": 0.5, "mmlu": 0.5})
                self.assertEqual(calls, [(["piqa"], 0, None, 431), (["mmlu"], 5, None, 503)])
                details = read_json(path)
                self.assertEqual(details["status"], "complete")
                self.assertIsNone(details["limit"])
                self.assertEqual(details["runs"][1]["n_samples"]["mmlu"]["effective"], 503)
                self.assertTrue(model.training)

    def test_default_and_zero_utility_limit_include_all_eligible_lines(self):
        # More than the former 64-line cap; the existing length filter remains.
        eligible = [f"Document {i}: " + "text " * 20 for i in range(131)]
        dataset = {"text": ["", "short", *eligible]}
        parser = compare_unlearning.build_parser()
        for flags in ([], ["--n_utility", "0"], ["--n_utility", "full"]):
            args = parser.parse_args(flags)
            with self.subTest(flags=flags), patch.object(compare_unlearning, "load_dataset", return_value=dataset):
                self.assertEqual(compare_unlearning.load_utility_texts(args.n_utility),
                                 [text.strip() for text in eligible])
        args = parser.parse_args(["--n_utility", "3", "--lm_eval_limit", "20"])
        self.assertEqual(args.lm_eval_limit, 20)
        with patch.object(compare_unlearning, "load_dataset", return_value=dataset):
            self.assertEqual(compare_unlearning.load_utility_texts(args.n_utility),
                             [text.strip() for text in eligible[:3]])


if __name__ == "__main__":
    unittest.main()
