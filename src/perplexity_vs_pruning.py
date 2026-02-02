import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from pruner import prune_attn_w_column


def calculate_perplexity(model, tokenizer, dataset, max_samples=100, max_length=512):
    """Calculate perplexity on a dataset.

    Args:
        model: The language model
        tokenizer: The tokenizer
        dataset: Dataset to evaluate on
        max_samples: Maximum number of samples to evaluate
        max_length: Maximum sequence length

    Returns:
        float: Perplexity score
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for idx, example in enumerate(tqdm(dataset, desc="Calculating perplexity")):
            if idx >= max_samples:
                break

            # Get text based on dataset structure
            if 'text' in example:
                text = example['text']
            elif 'input' in example:
                text = example['input']
            else:
                # Try to find any text field
                text = str(list(example.values())[0])

            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Get model output
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            # Accumulate loss weighted by number of tokens
            num_tokens = inputs["input_ids"].size(1)
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    # Calculate average loss and perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return perplexity


def plot_ppl_vs_pruning(results, save_path="ppl_vs_pruning.png"):
    """Plot perplexity vs pruning ratio in IEEE standard format.

    Args:
        results: List of dicts with 'prune_ratio' and 'perplexity' keys
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
    perplexities = [r['perplexity'] for r in results]

    # Plot
    ax.plot(prune_ratios, perplexities, marker='o', linewidth=1.5,
            markersize=5, color='#0173B2', label='Perplexity')

    # Add horizontal line at baseline perplexity
    if len(results) > 0:
        baseline_ppl = results[0]['perplexity']
        ax.axhline(y=baseline_ppl, color='#DE8F05',
                   linestyle='--', linewidth=1.5, label='Baseline')

    # Customize plot
    ax.set_xlabel('Pruning Ratio (%)')
    ax.set_ylabel('Perplexity')
    ax.set_title('Perplexity vs. Pruning Ratio')
    ax.grid(True, linestyle=':', alpha=0.6, linewidth=0.5)
    ax.legend(loc='best', frameon=True, edgecolor='black', fancybox=False)
    ax.set_xlim(left=0)

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
    """Main function to evaluate perplexity vs pruning ratio."""

    # Model configuration
    model_name = "EleutherAI/pythia-2.8b"

    # Pruning ratios to test (10%, 15%, 20%, 25%, 30%)
    prune_ratios = [0.10, 0.15, 0.20, 0.25, 0.30]

    print("="*50)
    print("Perplexity vs Pruning Ratio Experiment")
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

    # Load evaluation dataset (WikiText-2 is commonly used for perplexity)
    print("\n" + "="*50)
    print("Loading Evaluation Dataset...")
    print("="*50)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # Filter out empty examples
    dataset = dataset.filter(lambda x: len(x['text'].strip()) > 0)
    print(f"Loaded {len(dataset)} test samples")

    # Store results
    results = []

    # Evaluate baseline (0% pruning)
    print("\n" + "="*50)
    print("Baseline (No Pruning)")
    print("="*50)
    baseline_ppl = calculate_perplexity(model, tokenizer, dataset, max_samples=100)
    print(f"Baseline Perplexity: {baseline_ppl:.4f}")
    results.append({'prune_ratio': 0.0, 'perplexity': baseline_ppl})

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

        # Evaluate perplexity
        ppl = calculate_perplexity(model, tokenizer, dataset, max_samples=100)
        print(f"Perplexity: {ppl:.4f}")
        print(f"Change from baseline: {ppl - baseline_ppl:+.4f} ({((ppl - baseline_ppl) / baseline_ppl * 100):+.2f}%)")

        results.append({'prune_ratio': prune_ratio, 'perplexity': ppl})

    # Plot results
    print("\n" + "="*50)
    print("Generating IEEE Standard Plot...")
    print("="*50)
    plot_ppl_vs_pruning(results, save_path="ppl_vs_pruning.png")

    # Print summary table
    print("\n" + "="*50)
    print("Summary of Results")
    print("="*50)
    print(f"{'Pruning Ratio':<15} {'Perplexity':<12} {'Change from Baseline':<25}")
    print("-"*52)
    for r in results:
        change = r['perplexity'] - baseline_ppl
        change_pct = (change / baseline_ppl * 100) if baseline_ppl > 0 else 0
        print(f"{r['prune_ratio']*100:>5.0f}%          {r['perplexity']:>8.4f}     {change:+8.4f} ({change_pct:+6.2f}%)")


if __name__ == "__main__":
    main()
