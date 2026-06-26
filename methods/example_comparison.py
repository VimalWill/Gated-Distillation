"""
Example: Running all unlearning methods for comparison.

This script shows how to integrate DEL and SPE-Unlearn with your existing
pruning methods for comprehensive benchmarking.
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from methods import DELUnlearning, SPEUnlearning
from methods.comparison import UnlearningComparison, default_accuracy_fn


def create_dummy_loader(batch_size: int = 32, num_batches: int = 10):
    """Create dummy dataloader for testing. Replace with real data."""
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples):
            self.num_samples = num_samples

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            return {
                "input_ids": torch.randint(0, 1000, (128,)),
                "attention_mask": torch.ones(128, dtype=torch.long),
                "labels": torch.randint(0, 10, ()),
            }

    dataset = DummyDataset(batch_size * num_batches)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def create_dummy_model():
    """Create dummy model for testing. Replace with your actual model."""
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=10
    )


def method_del(model, forget_loader, retain_loader, **kwargs):
    """Wrapper for DEL unlearning method."""
    unlearner = DELUnlearning(model)
    mask = unlearner.unlearn(
        forget_loader,
        retain_loader,
        budget_alpha=kwargs.get("budget_alpha", 0.25),
        finetune_lr=kwargs.get("finetune_lr", 0.01),
        finetune_epochs=kwargs.get("finetune_epochs", 1),
    )
    return mask


def method_spe(model, forget_loader, retain_loader, **kwargs):
    """Wrapper for SPE-Unlearn method."""
    unlearner = SPEUnlearning(model)
    mask = unlearner.unlearn(
        retain_loader,
        forget_loader,
        sparsity=kwargs.get("sparsity", 0.9),
        learning_rate=kwargs.get("spe_lr", 1e-6),
    )
    return mask


def example_minimal_comparison():
    """Minimal example: compare DEL vs SPE on dummy data."""
    print("Loading model...")
    model = create_dummy_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("Creating dummy data...")
    forget_loader = create_dummy_loader(batch_size=8, num_batches=5)
    retain_loader = create_dummy_loader(batch_size=8, num_batches=10)
    test_loader = create_dummy_loader(batch_size=8, num_batches=5)

    # Define methods to compare
    methods = {
        "DEL (α=0.25)": lambda m, **kw: method_del(
            m, forget_loader, retain_loader, budget_alpha=0.25, **kw
        ),
        "SPE (S=0.9)": lambda m, **kw: method_spe(
            m, forget_loader, retain_loader, sparsity=0.9, **kw
        ),
    }

    # Run comparison
    comparison = UnlearningComparison(model, device)
    results = comparison.run_comparison(
        methods,
        forget_loader,
        retain_loader,
        test_loader,
        default_accuracy_fn,
    )

    # Print summary
    comparison.print_summary(results)


def example_hyperparameter_sweep():
    """Example: sweep DEL budget and SPE sparsity parameters."""
    print("Loading model...")
    model = create_dummy_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("Creating dummy data...")
    forget_loader = create_dummy_loader(batch_size=8, num_batches=5)
    retain_loader = create_dummy_loader(batch_size=8, num_batches=10)
    test_loader = create_dummy_loader(batch_size=8, num_batches=5)

    budgets = [0.15, 0.25, 0.35]
    sparsities = [0.85, 0.90, 0.95]

    all_results = {}

    for budget in budgets:
        for sparsity in sparsities:
            methods = {
                f"DEL (α={budget})": lambda m, **kw: method_del(
                    m, forget_loader, retain_loader, budget_alpha=budget, **kw
                ),
                f"SPE (S={sparsity})": lambda m, **kw: method_spe(
                    m, forget_loader, retain_loader, sparsity=sparsity, **kw
                ),
            }

            comparison = UnlearningComparison(model, device)
            results = comparison.run_comparison(
                methods,
                forget_loader,
                retain_loader,
                test_loader,
                default_accuracy_fn,
            )
            all_results.update(results)

    # Summary
    print("\n" + "="*80)
    print("HYPERPARAMETER SWEEP SUMMARY")
    print("="*80)
    for method_name, result in all_results.items():
        print(
            f"{method_name:<30} Forget: {result.forget_accuracy:.4f} | "
            f"Retain: {result.retain_accuracy:.4f} | Test: {result.model_accuracy:.4f}"
        )


if __name__ == "__main__":
    print("DEL and SPE-Unlearn Comparison Example\n")

    # Run minimal comparison
    print("="*80)
    print("MINIMAL COMPARISON (DEL vs SPE)")
    print("="*80)
    example_minimal_comparison()

    # Uncomment to run hyperparameter sweep
    # print("\n" + "="*80)
    # print("HYPERPARAMETER SWEEP")
    # print("="*80)
    # example_hyperparameter_sweep()
