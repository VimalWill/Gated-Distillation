import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt


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

    # ROC-AUC
    auc = roc_auc_score(labels, scores)

    # Min-K% scores split by member/non-member
    member_scores = [s for s, l in zip(scores, labels) if l == 1]
    nonmember_scores = [s for s, l in zip(scores, labels) if l == 0]
    mink_member = np.mean(member_scores)
    mink_nonmember = np.mean(nonmember_scores)
    mink_gap = mink_member - mink_nonmember

    print(f"ROC-AUC: {auc:.4f}")
    print(f"Min-K% member:     {mink_member:.4f}")
    print(f"Min-K% non-member: {mink_nonmember:.4f}")
    print(f"Min-K% gap:        {mink_gap:.4f}")
    print(f"Total samples: {len(labels)}")
    print(f"Memorized (label=1): {sum(labels)}")
    print(f"Non-memorized (label=0): {len(labels) - sum(labels)}")

    return {
        'auc': auc,
        'mink_member': mink_member,
        'mink_nonmember': mink_nonmember,
        'mink_gap': mink_gap,
    }


def plot_pruning_vs_memorization(results, save_path="memorization_pruning_results.png"):
    """Plot ROC-AUC and Min-K% Gap on single plot with dual y-axes."""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.titlesize'] = 11

    fig, ax1 = plt.subplots(figsize=(7.16, 3.5))

    prune_ratios = [r['prune_ratio'] * 100 for r in results]
    auc_scores = [r['auc'] for r in results]
    mink_gap = [r['mink_gap'] for r in results]

    # Left y-axis: ROC-AUC
    color_auc = '#0173B2'
    ax1.plot(prune_ratios, auc_scores, marker='o', linewidth=1.5,
             markersize=5, color=color_auc, label='ROC-AUC')
    ax1.set_xlabel('Pruning Ratio (%)')
    ax1.set_ylabel('ROC-AUC Score', color=color_auc)
    ax1.tick_params(axis='y', labelcolor=color_auc)
    ax1.set_xlim(left=0)
    ax1.set_ylim([0.4, 1.0])
    ax1.grid(True, linestyle=':', alpha=0.6, linewidth=0.5)

    # Right y-axis: Min-K% Gap
    ax2 = ax1.twinx()
    color_gap = '#D55E00'
    ax2.plot(prune_ratios, mink_gap, marker='^', linewidth=1.5,
             markersize=5, color=color_gap, linestyle='--', label='Min-K% Gap')
    ax2.set_ylabel('Min-K% Gap (member - non-member)', color=color_gap)
    ax2.tick_params(axis='y', labelcolor=color_gap)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best',
               frameon=True, edgecolor='black', fancybox=False)

    ax1.set_title('Memorization Detection vs. Pruning Ratio')
    plt.tight_layout()

    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")

    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {pdf_path}")

    plt.close()

def main():
    LENGTH = 64
    dataset = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{LENGTH}")

    original_model_name = "EleutherAI/pythia-2.8b"
    trained_model_path = "/home/u32/vimalwilliam/Gated-Distillation/src/trained_model/"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Evaluate original model ---
    print("="*50)
    print("Original Model (pythia-2.8b)")
    print("="*50)
    tokenizer = AutoTokenizer.from_pretrained(original_model_name)
    model = AutoModelForCausalLM.from_pretrained(original_model_name, dtype=torch.float16)
    model = model.to(device)
    model.eval()
    baseline = estimate_memorization(model, dataset, tokenizer)
    del model
    torch.cuda.empty_cache()

    # --- Evaluate unlearned model ---
    print("\n" + "="*50)
    print("Unlearned Model (gradient ascent on min token)")
    print("="*50)
    tokenizer = AutoTokenizer.from_pretrained(trained_model_path)
    model = AutoModelForCausalLM.from_pretrained(trained_model_path, dtype=torch.float16)
    model = model.to(device)
    model.eval()
    unlearned = estimate_memorization(model, dataset, tokenizer)
    del model
    torch.cuda.empty_cache()

    # --- Comparison ---
    print("\n" + "="*50)
    print("Comparison")
    print("="*50)
    print(f"{'Metric':<20} {'Original':>10} {'Unlearned':>10} {'Delta':>10}")
    print("-"*52)
    print(f"{'ROC-AUC':<20} {baseline['auc']:>10.4f} {unlearned['auc']:>10.4f} {unlearned['auc']-baseline['auc']:>+10.4f}")
    print(f"{'MinK member':<20} {baseline['mink_member']:>10.4f} {unlearned['mink_member']:>10.4f} {unlearned['mink_member']-baseline['mink_member']:>+10.4f}")
    print(f"{'MinK non-member':<20} {baseline['mink_nonmember']:>10.4f} {unlearned['mink_nonmember']:>10.4f} {unlearned['mink_nonmember']-baseline['mink_nonmember']:>+10.4f}")
    print(f"{'MinK gap':<20} {baseline['mink_gap']:>10.4f} {unlearned['mink_gap']:>10.4f} {unlearned['mink_gap']-baseline['mink_gap']:>+10.4f}")


if __name__ == "__main__":
    main()
        
