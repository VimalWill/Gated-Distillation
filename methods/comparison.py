"""Utilities for comparing unlearning methods."""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class UnlearningResult:
    """Results from unlearning evaluation."""
    method_name: str
    forget_accuracy: float
    retain_accuracy: float
    model_accuracy: float
    params_modified: int
    timestamp: datetime


class UnlearningComparison:
    """Framework for comparing unlearning methods."""

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        """
        Initialize comparison framework.

        Args:
            model: Original model to compare methods on
            device: Computation device
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        self.results: List[UnlearningResult] = []

    def evaluate_forget_accuracy(
        self, forget_loader, accuracy_fn: Callable
    ) -> float:
        """
        Evaluate accuracy on forget set (lower = better unlearning).

        Args:
            forget_loader: DataLoader with data to forget
            accuracy_fn: Function computing accuracy

        Returns:
            Accuracy on forget set
        """
        self.model.eval()
        return accuracy_fn(self.model, forget_loader, self.device)

    def evaluate_retain_accuracy(
        self, retain_loader, accuracy_fn: Callable
    ) -> float:
        """
        Evaluate accuracy on retain set (higher = better retention).

        Args:
            retain_loader: DataLoader with data to retain
            accuracy_fn: Function computing accuracy

        Returns:
            Accuracy on retain set
        """
        self.model.eval()
        return accuracy_fn(self.model, retain_loader, self.device)

    def evaluate_test_accuracy(
        self, test_loader, accuracy_fn: Callable
    ) -> float:
        """
        Evaluate accuracy on test set (higher = better generalization).

        Args:
            test_loader: DataLoader with test data
            accuracy_fn: Function computing accuracy

        Returns:
            Test accuracy
        """
        self.model.eval()
        return accuracy_fn(self.model, test_loader, self.device)

    def count_modified_params(self, mask: Dict[str, torch.Tensor]) -> int:
        """Count number of parameters marked for modification."""
        total = 0
        for param_mask in mask.values():
            if isinstance(param_mask, torch.Tensor):
                total += param_mask.sum().item()
        return int(total)

    def run_comparison(
        self,
        methods: Dict[str, Callable],
        forget_loader,
        retain_loader,
        test_loader,
        accuracy_fn: Callable,
        **method_kwargs,
    ) -> Dict[str, UnlearningResult]:
        """
        Run all unlearning methods and compare results.

        Args:
            methods: Dict of method_name -> unlearning_callable
            forget_loader: Data to forget
            retain_loader: Data to retain
            test_loader: Test set
            accuracy_fn: Function to compute accuracy
            **method_kwargs: Arguments passed to each method

        Returns:
            Dictionary of results per method
        """
        results = {}

        for method_name, method in methods.items():
            print(f"\n{'='*60}")
            print(f"Running {method_name}...")
            print(f"{'='*60}")

            # Create fresh model copy for each method
            model_copy = self._copy_model()

            # Run unlearning
            mask = method(model_copy, **method_kwargs)

            # Evaluate
            forget_acc = self.evaluate_forget_accuracy(forget_loader, accuracy_fn)
            retain_acc = self.evaluate_retain_accuracy(retain_loader, accuracy_fn)
            test_acc = self.evaluate_test_accuracy(test_loader, accuracy_fn)
            params_modified = self.count_modified_params(mask)

            result = UnlearningResult(
                method_name=method_name,
                forget_accuracy=forget_acc,
                retain_accuracy=retain_acc,
                model_accuracy=test_acc,
                params_modified=params_modified,
                timestamp=datetime.now(),
            )
            results[method_name] = result

            print(f"\n{method_name} Results:")
            print(f"  Forget Accuracy: {forget_acc:.4f} (lower=better)")
            print(f"  Retain Accuracy: {retain_acc:.4f} (higher=better)")
            print(f"  Test Accuracy:   {test_acc:.4f} (higher=better)")
            print(f"  Params Modified: {params_modified}")

        self.results = list(results.values())
        return results

    def print_summary(self, results: Dict[str, UnlearningResult]):
        """Print summary table of all methods."""
        print(f"\n{'='*80}")
        print("UNLEARNING COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(
            f"{'Method':<25} {'Forget Acc':<12} {'Retain Acc':<12} "
            f"{'Test Acc':<12} {'Params Mod':<12}"
        )
        print(f"{'-'*80}")

        for result in results.values():
            print(
                f"{result.method_name:<25} {result.forget_accuracy:<12.4f} "
                f"{result.retain_accuracy:<12.4f} {result.model_accuracy:<12.4f} "
                f"{result.params_modified:<12}"
            )

    def _copy_model(self) -> nn.Module:
        """Create a deep copy of the model."""
        import copy
        return copy.deepcopy(self.model)


def default_accuracy_fn(model: nn.Module, loader, device: torch.device) -> float:
    """Default accuracy function for text/NLP models."""
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            outputs = model(**{k: v for k, v in batch.items() if k != "labels"})
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            labels = batch.get("labels")

            if labels is None:
                continue

            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0.0
