import zlib
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


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


def calculate_perplexity(text, model, tokenizer, max_length=2048):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return torch.exp(outputs.loss).item()


def estimate_memorization(model, dataset, tokenizer, ref_model=None, ref_tokenizer=None):
    """Compute PPL, Lowercase, Zlib, Smaller Ref, and Min-K% memorization metrics.

    For each metric, higher score = more likely to be a member (memorized).
    ROC-AUC is computed per metric against ground-truth labels.
    """
    scores = {"PPL": [], "Lowercase": [], "Zlib": [], "Min-K%": []}
    if ref_model is not None:
        scores["Smaller Ref"] = []
    labels = []

    for ex in tqdm(dataset):
        text = ex["input"]

        ppl = calculate_perplexity(text, model, tokenizer)
        ppl_lower = calculate_perplexity(text.lower(), model, tokenizer)
        zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))
        log_probs = get_token_logprobs(text, tokenizer, model)
        mink = min_k_percent_score(log_probs, k=0.2)

        # Higher score = more likely member
        scores["PPL"].append(-np.log(ppl))
        scores["Lowercase"].append(np.log(ppl_lower) / np.log(ppl))
        scores["Zlib"].append(zlib_entropy / np.log(ppl))
        scores["Min-K%"].append(mink)

        if ref_model is not None:
            ppl_small = calculate_perplexity(text, ref_model, ref_tokenizer)
            scores["Smaller Ref"].append(np.log(ppl_small) / np.log(ppl))

        labels.append(ex["label"])

    results = {}
    for metric, s in scores.items():
        results[metric] = roc_auc_score(labels, s) if len(set(labels)) > 1 else 0.5

    print(f"\n{'Metric':<15} {'ROC-AUC':>10}")
    print("-" * 27)
    for metric, auc in results.items():
        print(f"{metric:<15} {auc:>10.4f}")

    return results


def main():
    LENGTH = 64
    dataset = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{LENGTH}")

    original_model_name = "EleutherAI/pythia-2.8b"
    small_model_name    = "EleutherAI/pythia-160m"
    trained_model_path  = "trained_model"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load small reference model once (shared across both evaluations)
    print(f"\nLoading smaller reference model ({small_model_name})...")
    ref_tokenizer = AutoTokenizer.from_pretrained(small_model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(small_model_name, torch_dtype=torch.float16)
    ref_model = ref_model.to(device)
    ref_model.eval()

    # --- Original model ---
    print("\n" + "="*50)
    print("Original Model (pythia-2.8b)")
    print("="*50)
    tokenizer = AutoTokenizer.from_pretrained(original_model_name)
    model = AutoModelForCausalLM.from_pretrained(original_model_name, torch_dtype=torch.float16)
    model = model.to(device)
    model.eval()
    baseline = estimate_memorization(model, dataset, tokenizer, ref_model, ref_tokenizer)
    del model
    torch.cuda.empty_cache()

    # --- Unlearned model ---
    print("\n" + "="*50)
    print("Unlearned Model")
    print("="*50)
    tokenizer = AutoTokenizer.from_pretrained(trained_model_path)
    model = AutoModelForCausalLM.from_pretrained(trained_model_path, torch_dtype=torch.float16)
    model = model.to(device)
    model.eval()
    unlearned = estimate_memorization(model, dataset, tokenizer, ref_model, ref_tokenizer)
    del model
    torch.cuda.empty_cache()

    # --- Comparison ---
    print("\n" + "="*50)
    print("Comparison (ROC-AUC per metric)")
    print("="*50)
    print(f"{'Metric':<15} {'Original':>10} {'Unlearned':>10} {'Delta':>10}")
    print("-" * 47)
    for metric in baseline:
        delta = unlearned[metric] - baseline[metric]
        print(f"{metric:<15} {baseline[metric]:>10.4f} {unlearned[metric]:>10.4f} {delta:>+10.4f}")


if __name__ == "__main__":
    main()
