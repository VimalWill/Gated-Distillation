# Unlearning Methods for Comparison

This directory contains implementations of two state-of-the-art machine unlearning methods for comparing with your gated distillation approach.

## Methods Included

### 1. DEL (Deletion by Example Localization)
**Paper:** "Improved Localized Machine Unlearning Through the Lens of Memorization"  
**Authors:** Torkzadehmahani et al.

#### Key Features
- Uses **weighted gradient criticality** (θ * ∇ℓ) to identify important parameters
- Channel/neuron-level granularity (more robust than parameter-level)
- **Reset + Finetune** unlearning strategy
- Optimal parameter budget: 20-30%

#### Usage
```python
from methods import DELUnlearning

unlearner = DELUnlearning(model, device=device)
mask = unlearner.unlearn(
    forget_loader,
    retain_loader,
    budget_alpha=0.25,        # Update 25% of params
    finetune_lr=0.01,
    finetune_epochs=1,
)
```

#### Configuration
- `budget_alpha`: Fraction of parameters to unlearn (0.2-0.3 optimal)
- `top_h`: Top-h scores per channel to average for criticality
- `finetune_lr`: Learning rate for finetuning phase
- `finetune_epochs`: Number of finetuning epochs on retain set

---

### 2. SPE-Unlearn (Structure-Aware Parameter-Efficient)
**Paper:** "Structure-Aware Parameter-Efficient Machine Unlearning on Transformer Models"

#### Key Features
- Uses **Fisher Information Matrix + forget gradients** for importance scoring
- Targets attention heads and FFN filters (structural units)
- **Sparse second-order update** for efficient unlearning
- Optimal sparsity: 90% (freeze 90%, update 10%)
- Enables successive unlearning on multiple requests

#### Usage
```python
from methods import SPEUnlearning

unlearner = SPEUnlearning(model, device=device)
mask = unlearner.unlearn(
    retain_loader,
    forget_loader,
    sparsity=0.9,             # Freeze 90%, update 10%
    learning_rate=1e-6,
)
```

#### Configuration
- `sparsity`: Fraction of parameters to freeze (0.85-0.95 optimal)
- `learning_rate`: Sparse second-order update rate (1e-6 to 1e-7)
- `target_ratio`: Alternative to sparsity; directly set update fraction

---

## Comparison Framework

Use `UnlearningComparison` for automated benchmarking:

```python
from methods.comparison import UnlearningComparison, default_accuracy_fn

comparison = UnlearningComparison(model, device=device)
results = comparison.run_comparison(
    methods={
        "DEL (α=0.25)": lambda m, **kw: method_del(m, forget_loader, retain_loader, **kw),
        "SPE (S=0.9)": lambda m, **kw: method_spe(m, forget_loader, retain_loader, **kw),
    },
    forget_loader=forget_loader,
    retain_loader=retain_loader,
    test_loader=test_loader,
    accuracy_fn=default_accuracy_fn,  # or your custom function
)

comparison.print_summary(results)
```

### Evaluation Metrics
- **Forget Accuracy**: Lower is better (successfully forgotten)
- **Retain Accuracy**: Higher is better (utility preserved)
- **Test Accuracy**: Higher is better (generalization)
- **Params Modified**: Number of parameters updated

---

## Quick Start

### 1. Basic Comparison
```bash
python methods/example_comparison.py
```

### 2. Custom Comparison
```python
from methods import DELUnlearning, SPEUnlearning

# Initialize unlearners
del_unlearner = DELUnlearning(model)
spe_unlearner = SPEUnlearning(model)

# Run unlearning
del_mask = del_unlearner.unlearn(forget_loader, retain_loader)
spe_mask = spe_unlearner.unlearn(retain_loader, forget_loader)

# Evaluate
forget_acc = evaluate(model, forget_loader)
retain_acc = evaluate(model, retain_loader)
```

---

## Integration with Existing Methods

To compare with your existing pruning methods (prune_l1_unstructured, prune_wanda, etc.):

```python
from src.pruner import prune_l1_unstructured, prune_wanda
from methods import DELUnlearning, SPEUnlearning

def method_pruning_l1(model, **kwargs):
    prune_l1_unstructured(model, prune_ratio=0.25)
    return {}  # No mask returned

methods = {
    "L1 Unstructured": lambda m, **kw: method_pruning_l1(m, **kw),
    "Wanda": lambda m, **kw: prune_wanda(m, **kw),
    "DEL": lambda m, **kw: method_del(m, forget_loader, retain_loader, **kw),
    "SPE": lambda m, **kw: method_spe(m, retain_loader, forget_loader, **kw),
}
```

---

## Key Differences: DEL vs SPE

| Aspect | DEL | SPE |
|--------|-----|-----|
| Localization | Data-dependent (memorization-informed) | Theory-driven (Taylor + FIM) |
| Granularity | Channels/neurons | Attention heads/FFN filters |
| Optimal Budget | 20-30% (update ratio) | 90% (freeze ratio) |
| Unlearning Strategy | Reset + Finetune | Sparse second-order update |
| Successive Requests | Not addressed | Memory-free & memory-aided |
| Scalability | Moderate | High (110M → 37K units) |
| Archicture | Vision + general | Transformers |

---

## References

1. **DEL:** Torkzadehmahani et al., "Improved Localized Machine Unlearning Through the Lens of Memorization" (OpenReview, NeurIPS 2024)

2. **SPE-Unlearn:** Anonymous, "Structure-Aware Parameter-Efficient Machine Unlearning on Transformer Models" (ICLR 2025 under review)

---

## Files

- `__init__.py` - Module exports
- `del_unlearning.py` - DEL implementation
- `spe_unlearning.py` - SPE-Unlearn implementation
- `comparison.py` - Benchmarking framework
- `example_comparison.py` - Example scripts and usage patterns
- `README.md` - This file
