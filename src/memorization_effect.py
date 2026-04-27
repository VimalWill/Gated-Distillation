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


def safe_auc(scores, labels):
    return roc_auc_score(labels, scores) if len(set(labels)) > 1 else 0.5


def estimate_memorization(model, dataset, tokenizer, ref_model=None, ref_tokenizer=None):
    """Compute memorization metrics and return a results dict with:
      roc_auc, mink_member, mink_nonmember, mink_gap,
      ppl_auc, lowercase_auc, zlib_auc, smaller_ref_auc
    """
    mink_scores, ppl_scores, lower_scores, zlib_scores, ref_scores = [], [], [], [], []
    labels = []

    for ex in tqdm(dataset):
        text = ex["input"]

        ppl       = calculate_perplexity(text, model, tokenizer)
        ppl_lower = calculate_perplexity(text.lower(), model, tokenizer)
        z         = len(zlib.compress(bytes(text, "utf-8")))
        log_probs = get_token_logprobs(text, tokenizer, model)
        mink      = min_k_percent_score(log_probs, k=0.2)

        mink_scores.append(mink)
        ppl_scores.append(-np.log(ppl))
        lower_scores.append(np.log(ppl_lower) / np.log(ppl))
        zlib_scores.append(z / np.log(ppl))

        if ref_model is not None:
            ppl_ref = calculate_perplexity(text, ref_model, ref_tokenizer)
            ref_scores.append(np.log(ppl_ref) / np.log(ppl))

        labels.append(ex["label"])

    labels     = np.array(labels)
    mink_arr   = np.array(mink_scores)
    members    = mink_arr[labels == 1]
    nonmembers = mink_arr[labels == 0]

    return {
        "roc_auc":         safe_auc(mink_arr,    labels),
        "mink_member":     float(members.mean())    if len(members)    else 0.0,
        "mink_nonmember":  float(nonmembers.mean()) if len(nonmembers) else 0.0,
        "mink_gap":        float(members.mean() - nonmembers.mean()) if len(members) and len(nonmembers) else 0.0,
        "ppl_auc":         safe_auc(ppl_scores,   labels),
        "lowercase_auc":   safe_auc(lower_scores,  labels),
        "zlib_auc":        safe_auc(zlib_scores,   labels),
        "smaller_ref_auc": safe_auc(ref_scores,    labels) if ref_scores else None,
    }


COLS = [
    ("ROC-AUC",   "roc_auc"),
    ("MK-Mem",    "mink_member"),
    ("MK-Non",    "mink_nonmember"),
    ("MK-Gap",    "mink_gap"),
    ("PPL",       "ppl_auc"),
    ("Lowercase", "lowercase_auc"),
    ("Zlib",      "zlib_auc"),
    ("SmRef",     "smaller_ref_auc"),
]


def print_table(results):
    name_w = max(len(k) for k in results) + 2
    header = f"{'Model':<{name_w}}" + "".join(f"{h:>10}" for h, _ in COLS)
    sep    = "=" * len(header)
    print(f"\n{sep}\n{header}\n{'-'*len(header)}")
    for model_name, r in results.items():
        row = f"{model_name:<{name_w}}"
        for _, key in COLS:
            v = r.get(key)
            row += f"{v:>10.4f}" if v is not None else f"{'N/A':>10}"
        print(row)
    print(sep)


def print_delta_table(baseline, unlearned):
    print(f"\n{'Metric':<15} {'Original':>10} {'Unlearned':>10} {'Delta':>10}")
    print("-" * 47)
    for header, key in COLS:
        b = baseline.get(key)
        u = unlearned.get(key)
        if b is not None and u is not None:
            print(f"{header:<15} {b:>10.4f} {u:>10.4f} {u - b:>+10.4f}")
        else:
            print(f"{header:<15} {'N/A':>10} {'N/A':>10} {'N/A':>10}")


def main():
    LENGTH = 64
    dataset = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{LENGTH}")

    original_model_name = "EleutherAI/pythia-2.8b"
    small_model_name    = "EleutherAI/pythia-160m"
    trained_model_path  = "trained_model"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    # --- Results ---
    print("\n" + "="*50)
    print("Full Metrics Table")
    print_table({"Original": baseline, "Unlearned": unlearned})

    print("\n" + "="*50)
    print("Delta (Unlearned − Original)")
    print("="*50)
    print_delta_table(baseline, unlearned)


if __name__ == "__main__":
    main()
