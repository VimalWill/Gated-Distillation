"""
Layer-wise Pruning Analysis: ROC-AUC (Memorization) and Accuracy per Layer.

This script prunes each layer individually at different ratios (10%, 15%, 20%)
and measures both memorization detection (ROC-AUC) and task accuracy.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Import attention module type
try:
    from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention
except ImportError:
    GPTNeoXAttention = None


def prune_single_layer(model, layer_idx, prune_ratio):
    """Prune only a specific layer's attention weights.

    Args:
        model: The model to prune
        layer_idx: Index of the layer to prune (0-indexed)
        prune_ratio: Fraction of columns to prune
    """
    target_name = f"gpt_neox.layers.{layer_idx}.attention"

    for name, module in model.named_modules():
        if name != target_name:
            continue

        if not hasattr(module, 'query_key_value'):
            print(f"Warning: {name} doesn't have query_key_value")
            return

        qkv_weight = module.query_key_value.weight.data
        hidden_size = qkv_weight.size(0) // 3

        # Get Q, K, V weights
        q_weight = qkv_weight[:hidden_size, :]
        k_weight = qkv_weight[hidden_size:2*hidden_size, :]
        v_weight = qkv_weight[2*hidden_size:, :]

        # Calculate importance and prune
        def sparse_weights(weight, prune_ratio):
            importance = torch.sum(torch.abs(weight), dim=0)
            num_cols = weight.size(1)
            num_prune = int(prune_ratio * num_cols)
            if num_prune == 0:
                return
            prune_indices = torch.topk(importance, num_prune, largest=False).indices
            weight[:, prune_indices] = 0

        sparse_weights(q_weight, prune_ratio)
        sparse_weights(k_weight, prune_ratio)
        sparse_weights(v_weight, prune_ratio)

        return True

    return False


def get_token_logprobs(text, tokenizer, model):
    """Get log probabilities for each token."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, :-1, :]
        labels = inputs["input_ids"][:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)

    return token_log_probs.squeeze(0)


def min_k_percent_score(token_log_probs, k=0.2):
    """Calculate Min-K% score for memorization detection."""
    n = token_log_probs.numel()
    k_n = max(1, int(n * k))
    lowest = torch.topk(token_log_probs, k_n, largest=False).values
    return lowest.mean().item()


def calculate_roc_auc(model, tokenizer, dataset, max_samples=200):
    """Calculate ROC-AUC for memorization detection."""
    model.eval()
    scores = []
    labels = []

    with torch.no_grad():
        for idx, ex in enumerate(tqdm(dataset, desc="Calculating ROC-AUC", leave=False)):
            if idx >= max_samples:
                break
            log_probs = get_token_logprobs(ex["input"], tokenizer, model)
            score = min_k_percent_score(log_probs, k=0.2)
            scores.append(score)
            labels.append(ex["label"])

    if len(set(labels)) < 2:
        return 0.5  # Can't compute AUC with single class

    auc = roc_auc_score(labels, scores)
    return auc


def calculate_accuracy(model, tokenizer, dataset, max_samples=200):
    """Calculate accuracy on LAMBADA dataset."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for idx, example in enumerate(tqdm(dataset, desc="Calculating accuracy", leave=False)):
            if idx >= max_samples:
                break

            text = example['text']
            full_tokens = tokenizer(text, return_tensors="pt")
            full_ids = full_tokens['input_ids'][0]

            if len(full_ids) < 2:
                continue

            context_ids = full_ids[:-1].unsqueeze(0).to(model.device)
            target_id = full_ids[-1].item()

            outputs = model(input_ids=context_ids)
            logits = outputs.logits[0, -1, :]
            predicted_id = logits.argmax().item()

            if predicted_id == target_id:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0


def plot_layer_analysis(results, save_path="layer_analysis.png"):
    """Plot ROC-AUC and Accuracy vs Layer for different pruning ratios."""

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 10

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Colors for different pruning ratios
    colors = {'10': '#0173B2', '15': '#DE8F05', '20': '#029E73'}
    markers = {'10': 'o', '15': 's', '20': '^'}

    # Extract data
    layers = sorted(set(r['layer'] for r in results))
    prune_ratios = sorted(set(r['prune_ratio'] for r in results))

    # Plot ROC-AUC
    ax1 = axes[0]
    for pr in prune_ratios:
        pr_results = [r for r in results if r['prune_ratio'] == pr]
        pr_results.sort(key=lambda x: x['layer'])
        layer_vals = [r['layer'] for r in pr_results]
        auc_vals = [r['roc_auc'] for r in pr_results]
        ax1.plot(layer_vals, auc_vals, marker=markers[str(int(pr*100))],
                 linewidth=1.5, markersize=5, color=colors[str(int(pr*100))],
                 label=f'{int(pr*100)}% pruning')

    # Add baseline
    baseline_auc = results[0].get('baseline_auc', 0.5)
    ax1.axhline(y=baseline_auc, color='gray', linestyle='--', linewidth=1.5, label='Baseline')

    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('ROC-AUC Score')
    ax1.set_title('Memorization Detection (ROC-AUC) per Layer')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='best')
    ax1.set_ylim([0.4, 1.0])

    # Plot Accuracy
    ax2 = axes[1]
    for pr in prune_ratios:
        pr_results = [r for r in results if r['prune_ratio'] == pr]
        pr_results.sort(key=lambda x: x['layer'])
        layer_vals = [r['layer'] for r in pr_results]
        acc_vals = [r['accuracy'] * 100 for r in pr_results]
        ax2.plot(layer_vals, acc_vals, marker=markers[str(int(pr*100))],
                 linewidth=1.5, markersize=5, color=colors[str(int(pr*100))],
                 label=f'{int(pr*100)}% pruning')

    # Add baseline
    baseline_acc = results[0].get('baseline_acc', 0.5) * 100
    ax2.axhline(y=baseline_acc, color='gray', linestyle='--', linewidth=1.5, label='Baseline')

    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('LAMBADA Accuracy per Layer')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='best')
    ax2.set_ylim([0, 100])

    plt.tight_layout()

    # Save
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")

    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {pdf_path}")

    plt.close()


def plot_heatmap(results, save_path="layer_heatmap.png"):
    """Plot heatmaps for ROC-AUC and Accuracy changes."""

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Get unique layers and prune ratios
    layers = sorted(set(r['layer'] for r in results))
    prune_ratios = sorted(set(r['prune_ratio'] for r in results))

    baseline_auc = results[0].get('baseline_auc', 0.5)
    baseline_acc = results[0].get('baseline_acc', 0.5)

    # Create matrices
    auc_matrix = np.zeros((len(prune_ratios), len(layers)))
    acc_matrix = np.zeros((len(prune_ratios), len(layers)))

    for r in results:
        li = layers.index(r['layer'])
        pi = prune_ratios.index(r['prune_ratio'])
        auc_matrix[pi, li] = r['roc_auc'] - baseline_auc  # Change from baseline
        acc_matrix[pi, li] = (r['accuracy'] - baseline_acc) * 100  # Change from baseline

    # ROC-AUC heatmap
    im1 = axes[0].imshow(auc_matrix, cmap='RdYlGn', aspect='auto', vmin=-0.2, vmax=0.2)
    axes[0].set_xticks(range(len(layers)))
    axes[0].set_xticklabels(layers)
    axes[0].set_yticks(range(len(prune_ratios)))
    axes[0].set_yticklabels([f'{int(p*100)}%' for p in prune_ratios])
    axes[0].set_xlabel('Layer Index')
    axes[0].set_ylabel('Pruning Ratio')
    axes[0].set_title('ROC-AUC Change from Baseline')
    plt.colorbar(im1, ax=axes[0])

    # Accuracy heatmap
    im2 = axes[1].imshow(acc_matrix, cmap='RdYlGn', aspect='auto', vmin=-30, vmax=10)
    axes[1].set_xticks(range(len(layers)))
    axes[1].set_xticklabels(layers)
    axes[1].set_yticks(range(len(prune_ratios)))
    axes[1].set_yticklabels([f'{int(p*100)}%' for p in prune_ratios])
    axes[1].set_xlabel('Layer Index')
    axes[1].set_ylabel('Pruning Ratio')
    axes[1].set_title('Accuracy Change from Baseline (%)')
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()

    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {save_path}")

    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {pdf_path}")

    plt.close()


def main():
    """Main function for layer-wise pruning analysis."""

    # Configuration
    model_name = "EleutherAI/pythia-2.8b"

    # Layers to analyze (Pythia-2.8B has 32 layers, indices 0-31)
    # You can change this range
    layer_start = 0
    layer_end = 32  # Exclusive, so this will do layers 0-31

    # Pruning ratios to test
    prune_ratios = [0.10, 0.15, 0.20]

    # Sample sizes (reduce for faster testing)
    max_samples_auc = 200
    max_samples_acc = 200

    print("="*60)
    print("Layer-wise Pruning Analysis")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Layers: {layer_start} to {layer_end-1}")
    print(f"Pruning ratios: {[f'{p*100:.0f}%' for p in prune_ratios]}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load datasets
    print("\nLoading datasets...")
    wikimia_dataset = load_dataset("swj0419/WikiMIA", split="WikiMIA_length64")
    lambada_dataset = load_dataset("lambada", split="test")
    print(f"WikiMIA samples: {len(wikimia_dataset)}")
    print(f"LAMBADA samples: {len(lambada_dataset)}")

    # Get baseline metrics
    print("\n" + "="*60)
    print("Computing Baseline Metrics...")
    print("="*60)

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16)
    model = model.to(device)
    model.eval()

    baseline_auc = calculate_roc_auc(model, tokenizer, wikimia_dataset, max_samples=max_samples_auc)
    baseline_acc = calculate_accuracy(model, tokenizer, lambada_dataset, max_samples=max_samples_acc)

    print(f"Baseline ROC-AUC: {baseline_auc:.4f}")
    print(f"Baseline Accuracy: {baseline_acc*100:.2f}%")

    del model
    torch.cuda.empty_cache()

    # Store results
    results = []

    # Analyze each layer
    total_experiments = (layer_end - layer_start) * len(prune_ratios)
    experiment_num = 0

    for layer_idx in range(layer_start, layer_end):
        for prune_ratio in prune_ratios:
            experiment_num += 1
            print("\n" + "="*60)
            print(f"Experiment {experiment_num}/{total_experiments}")
            print(f"Layer {layer_idx}, Pruning {prune_ratio*100:.0f}%")
            print("="*60)

            # Reload fresh model
            model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16)
            model = model.to(device)
            model.eval()

            # Prune only this layer
            success = prune_single_layer(model, layer_idx, prune_ratio)
            if not success:
                print(f"Warning: Could not prune layer {layer_idx}")
                del model
                torch.cuda.empty_cache()
                continue

            # Evaluate
            roc_auc = calculate_roc_auc(model, tokenizer, wikimia_dataset, max_samples=max_samples_auc)
            accuracy = calculate_accuracy(model, tokenizer, lambada_dataset, max_samples=max_samples_acc)

            print(f"ROC-AUC: {roc_auc:.4f} (Δ: {roc_auc - baseline_auc:+.4f})")
            print(f"Accuracy: {accuracy*100:.2f}% (Δ: {(accuracy - baseline_acc)*100:+.2f}%)")

            results.append({
                'layer': layer_idx,
                'prune_ratio': prune_ratio,
                'roc_auc': roc_auc,
                'accuracy': accuracy,
                'baseline_auc': baseline_auc,
                'baseline_acc': baseline_acc
            })

            # Cleanup
            del model
            torch.cuda.empty_cache()

    # Generate plots
    print("\n" + "="*60)
    print("Generating Plots...")
    print("="*60)

    plot_layer_analysis(results, save_path="layer_analysis.png")
    plot_heatmap(results, save_path="layer_heatmap.png")

    # Print summary table
    print("\n" + "="*60)
    print("Summary Table")
    print("="*60)
    print(f"{'Layer':<8} {'Prune%':<8} {'ROC-AUC':<10} {'Δ AUC':<10} {'Acc%':<10} {'Δ Acc%':<10}")
    print("-"*56)

    for r in sorted(results, key=lambda x: (x['layer'], x['prune_ratio'])):
        delta_auc = r['roc_auc'] - baseline_auc
        delta_acc = (r['accuracy'] - baseline_acc) * 100
        print(f"{r['layer']:<8} {int(r['prune_ratio']*100):<8} {r['roc_auc']:.4f}     {delta_auc:+.4f}     {r['accuracy']*100:.2f}      {delta_acc:+.2f}")


if __name__ == "__main__":
    main()
