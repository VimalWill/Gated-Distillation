import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

from pruner import prune_attn_w_column


def calculate_accuracy(model, tokenizer, dataset, max_samples=100):
    """Calculate accuracy on LAMBADA dataset (next word prediction).

    Args:
        model: The language model
        tokenizer: The tokenizer
        dataset: LAMBADA dataset
        max_samples: Maximum number of samples to evaluate

    Returns:
        float: Accuracy score (0-1)
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for idx, example in enumerate(tqdm(dataset, desc="Calculating accuracy")):
            if idx >= max_samples:
                break

            # LAMBADA: predict the last word given context
            text = example['text']

            # Tokenize the full text
            full_tokens = tokenizer(text, return_tensors="pt")
            full_ids = full_tokens['input_ids'][0]

            if len(full_ids) < 2:
                continue

            # Use all but last token as context
            context_ids = full_ids[:-1].unsqueeze(0).to(model.device)
            target_id = full_ids[-1].item()

            # Get model prediction
            outputs = model(input_ids=context_ids)
            logits = outputs.logits[0, -1, :]  # Last token logits
            predicted_id = logits.argmax().item()

            if predicted_id == target_id:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def plot_accuracy_vs_pruning(results, save_path="accuracy_vs_pruning.png"):
    """Plot accuracy vs pruning ratio in IEEE standard format.

    Args:
        results: List of dicts with 'prune_ratio' and 'accuracy' keys
        save_path: Path to save the plot
    """
    # Set IEEE style parameters
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.titlesize'] = 11

    # Create figure (IEEE two-column width)
    fig, ax = plt.subplots(figsize=(7.16, 2.5))

    # Extract data
    prune_ratios = [r['prune_ratio'] * 100 for r in results]
    accuracies = [r['accuracy'] * 100 for r in results]

    # Plot
    ax.plot(prune_ratios, accuracies, marker='o', linewidth=1.5,
            markersize=5, color='#0173B2', label='Accuracy')

    # Add horizontal line at baseline accuracy
    if len(results) > 0:
        baseline_acc = results[0]['accuracy'] * 100
        ax.axhline(y=baseline_acc, color='#DE8F05',
                   linestyle='--', linewidth=1.5, label='Baseline')

    # Customize plot
    ax.set_xlabel('Pruning Ratio (%)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('LAMBADA Accuracy vs. Pruning Ratio')
    ax.grid(True, linestyle=':', alpha=0.6, linewidth=0.5)
    ax.legend(loc='best', frameon=True, edgecolor='black', fancybox=False)
    ax.set_xlim(left=0)
    ax.set_ylim([0, 100])

    plt.tight_layout()

    # Save as PNG
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")

    # Also save as PDF
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {pdf_path}")

    plt.close()


def main():
    """Main function to evaluate accuracy vs pruning ratio."""

    # Model configuration
    model_name = "EleutherAI/pythia-2.8b"

    # Pruning ratios to test (10%, 15%, 20%, 25%, 30%)
    prune_ratios = [0.10, 0.15, 0.20, 0.25, 0.30]

    print("="*50)
    print("Accuracy vs Pruning Ratio Experiment")
    print("="*50)
    print(f"Model: {model_name}")
    print(f"Pruning ratios: {[f'{p*100:.0f}%' for p in prune_ratios]}")

    # Load tokenizer and model
    print("\n" + "="*50)
    print("Loading Model and Tokenizer...")
    print("="*50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
    )
    model = model.to(device)
    model.eval()

    # Load evaluation dataset (LAMBADA - next word prediction)
    print("\n" + "="*50)
    print("Loading LAMBADA Dataset...")
    print("="*50)
    dataset = load_dataset("lambada", split="test")
    print(f"Loaded {len(dataset)} test samples")

    # Store results
    results = []

    # Evaluate baseline (0% pruning)
    print("\n" + "="*50)
    print("Baseline (No Pruning)")
    print("="*50)
    baseline_acc = calculate_accuracy(model, tokenizer, dataset, max_samples=500)
    print(f"Baseline Accuracy: {baseline_acc*100:.2f}%")
    results.append({'prune_ratio': 0.0, 'accuracy': baseline_acc})

    # Test different pruning ratios
    for prune_ratio in prune_ratios:
        print("\n" + "="*50)
        print(f"Pruning Ratio: {prune_ratio*100:.0f}%")
        print("="*50)

        # Reload model for independent pruning
        print("Reloading fresh model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
        )
        model = model.to(device)
        model.eval()

        # Apply pruning
        prune_attn_w_column(model, prune_ratio=prune_ratio)

        # Evaluate accuracy
        acc = calculate_accuracy(model, tokenizer, dataset, max_samples=500)
        print(f"Accuracy: {acc*100:.2f}%")
        print(f"Change from baseline: {(acc - baseline_acc)*100:+.2f}% (absolute)")

        results.append({'prune_ratio': prune_ratio, 'accuracy': acc})

    # Plot results
    print("\n" + "="*50)
    print("Generating IEEE Standard Plot...")
    print("="*50)
    plot_accuracy_vs_pruning(results, save_path="accuracy_vs_pruning.png")

    # Print summary table
    print("\n" + "="*50)
    print("Summary of Results")
    print("="*50)
    print(f"{'Pruning Ratio':<15} {'Accuracy (%)':<15} {'Change from Baseline':<25}")
    print("-"*55)
    for r in results:
        change = (r['accuracy'] - baseline_acc) * 100
        print(f"{r['prune_ratio']*100:>5.0f}%          {r['accuracy']*100:>8.2f}        {change:+8.2f}%")


if __name__ == "__main__":
    main()
