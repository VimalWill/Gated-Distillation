import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt

from pruner import prune_attn_w_column

def get_token_logprobs(text, tokenizer, model):
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
    n = token_log_probs.numel()
    k_n = max(1, int(n * k))
    lowest = torch.topk(token_log_probs, k_n, largest=False).values
    return lowest.mean().item()


def estimate_memorization(model, dataset, tokenizer):
    scores = []
    labels = []

    for ex in tqdm(dataset):
        log_probs = get_token_logprobs(ex["input"], tokenizer, model)
        score = min_k_percent_score(log_probs, k=0.2)
        scores.append(score)
        labels.append(ex["label"])

    auc = roc_auc_score(labels, scores)
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Total samples: {len(labels)}")
    print(f"Memorized (label=1): {sum(labels)}")
    print(f"Non-memorized (label=0): {len(labels) - sum(labels)}")

    return auc


def plot_pruning_vs_auc(results, save_path="memorization_pruning_results.png"):
    """Plot pruning ratio vs ROC-AUC in IEEE standard format.

    Args:
        results: List of dicts with 'prune_ratio' and 'auc' keys
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
    auc_scores = [r['auc'] for r in results]

    # Plot
    ax.plot(prune_ratios, auc_scores, marker='o', linewidth=1.5,
            markersize=5, color='#0173B2', label='ROC-AUC Score')

    # Add horizontal line at baseline AUC
    if len(results) > 0:
        baseline_auc = results[0]['auc']
        ax.axhline(y=baseline_auc, color='#DE8F05',
                   linestyle='--', linewidth=1.5, label='Baseline')

    # Customize plot
    ax.set_xlabel('Pruning Ratio (%)')
    ax.set_ylabel('ROC-AUC Score')
    ax.set_title('Memorization Detection (ROC-AUC) vs. Pruning Ratio')
    ax.grid(True, linestyle=':', alpha=0.6, linewidth=0.5)
    ax.legend(loc='best', frameon=True, edgecolor='black', fancybox=False)
    ax.set_xlim(left=0)
    ax.set_ylim([0, 1.0])

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
    LENGTH = 64
    dataset = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{LENGTH}")

    model_name = "EleutherAI/pythia-2.8b"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    # Store results for plotting
    results = []

    print("="*50)
    print("Baseline (No Pruning)")
    print("="*50)
    baseline_auc = estimate_memorization(model, dataset, tokenizer)
    results.append({'prune_ratio': 0.0, 'auc': baseline_auc})

    # Test different pruning ratios
    prune_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for prune_ratio in prune_ratios:
        print("\n" + "="*50)
        print(f"Pruning Ratio: {prune_ratio*100:.0f}%")
        print("="*50)

        # Apply pruning
        prune_attn_w_column(model, prune_ratio=prune_ratio)

        # Evaluate memorization
        auc = estimate_memorization(model, dataset, tokenizer)
        results.append({'prune_ratio': prune_ratio, 'auc': auc})

    # Plot results
    print("\n" + "="*50)
    print("Generating IEEE Standard Plot...")
    print("="*50)
    plot_pruning_vs_auc(results, save_path="memorization_pruning_results.png")

    # Print summary
    print("\n" + "="*50)
    print("Summary of Results")
    print("="*50)
    print(f"{'Pruning Ratio':<15} {'ROC-AUC':<10} {'Change from Baseline':<20}")
    print("-"*50)
    for r in results:
        change = r['auc'] - baseline_auc
        print(f"{r['prune_ratio']*100:>5.0f}%          {r['auc']:.4f}     {change:+.4f}")


if __name__ == "__main__":
    main()
        
