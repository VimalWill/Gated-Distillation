"""
Memorization-change metrics for de-memorization / unlearning methods.

These measure how much a model memorizes a set of "member" texts (e.g. the
forget set / training data) relative to "non-member" texts (held-out data),
and how that memorization CHANGES after a de-memorization method (pruning,
DEL, SPE-Unlearn, or the gated-distillation method) is applied.

The scoring primitives mirror `pruning_memorization_experiment.py` exactly, so
numbers are directly comparable across the pruning baselines and the methods/
comparison methods:

  - Min-K% Prob         (Shi et al., 2024)     -> ROC-AUC member vs non-member
  - Perplexity           (loss-based)          -> ROC-AUC
  - Lowercase ratio      (Carlini et al.)      -> ROC-AUC
  - Zlib entropy ratio   (Carlini et al.)      -> ROC-AUC
  - Smaller-Ref ratio    (reference model)     -> ROC-AUC

Interpretation for de-memorization: a model that has forgotten its members
becomes LESS able to tell members from non-members, so every ROC-AUC and the
Min-K gap should move toward the chance value (0.5 for AUC, 0 for the gap),
while member perplexity should RISE (the model is less confident on data it
used to memorize). Utility on non-members should be roughly preserved.
"""

import zlib
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

MAX_LENGTH = 2048
MIN_K_FRAC = 0.2


# ── Per-example scoring primitives (mirror pruning_memorization_experiment.py) ──

def get_token_logprobs(text, tokenizer, model, max_length=MAX_LENGTH):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits[:, :-1, :]
        labels = inputs["input_ids"][:, 1:]
    lp = F.log_softmax(logits.float(), dim=-1)
    return lp.gather(-1, labels.unsqueeze(-1)).squeeze(-1).squeeze(0)


def min_k_score(token_log_probs, k=MIN_K_FRAC):
    """Mean log-prob of the k-fraction least-likely tokens (higher => memorized)."""
    n = token_log_probs.numel()
    k_n = max(1, int(n * k))
    return torch.topk(token_log_probs, k_n, largest=False).values.mean().item()


def perplexity(text, model, tokenizer, max_length=MAX_LENGTH):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        loss = model(**inputs, labels=inputs["input_ids"]).loss
    return torch.exp(loss).item()


def safe_auc(scores, labels):
    scores = np.array(scores)
    labels = np.array(labels)
    if len(set(labels.tolist())) < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


# ── Full memorization panel for one model ─────────────────────────────────────

# Metric key -> whether a HIGHER value means MORE memorization/leakage.
# Used to sign the deltas so "improvement" (less memorization) is unambiguous.
_AUC_KEYS = ["roc_auc", "ppl_auc", "lowercase_auc", "zlib_auc", "smaller_ref_auc"]


def evaluate_memorization(
    model,
    tokenizer,
    dataset,
    ref_model=None,
    ref_tokenizer=None,
    k=MIN_K_FRAC,
    max_length=MAX_LENGTH,
    desc: Optional[str] = None,
) -> Dict[str, float]:
    """Compute the memorization panel for `model` over a member/non-member set.

    dataset: iterable of dicts with keys "input" (text) and "label"
             (1 = member / to-forget, 0 = non-member / held-out).
    ref_model/ref_tokenizer: optional smaller reference model for Smaller-Ref;
             if None, smaller_ref_auc is reported as 0.5 (chance).

    Returns a dict of AUCs, Min-K member/non-member means and gap, and the raw
    member/non-member mean perplexity (memorization strength in nats/PPL).
    """
    try:
        from tqdm import tqdm
        iterator = tqdm(dataset, desc=desc or "  memorization", leave=False)
    except Exception:
        iterator = dataset

    mink, ppl_s, lower_s, zlib_s, ref_s, labs, ppl_raw = [], [], [], [], [], [], []

    for ex in iterator:
        text = ex["input"]
        label = ex["label"]

        p = perplexity(text, model, tokenizer, max_length)
        p_low = perplexity(text.lower(), model, tokenizer, max_length)
        z = len(zlib.compress(text.encode("utf-8")))
        lp = get_token_logprobs(text, tokenizer, model, max_length)
        mk = min_k_score(lp, k=k)

        mink.append(mk)
        ppl_s.append(-np.log(p))                    # higher => more memorized
        lower_s.append(np.log(p_low) / np.log(p))
        zlib_s.append(z / np.log(p))
        labs.append(label)
        ppl_raw.append(p)

        if ref_model is not None:
            p_ref = perplexity(text, ref_model, ref_tokenizer or tokenizer, max_length)
            ref_s.append(np.log(p_ref) / np.log(p))
        else:
            ref_s.append(0.0)

    labs = np.array(labs)
    mink_a = np.array(mink)
    ppl_raw = np.array(ppl_raw)
    mem = mink_a[labs == 1]
    non = mink_a[labs == 0]
    ppl_mem = ppl_raw[labs == 1]
    ppl_non = ppl_raw[labs == 0]

    return {
        "roc_auc":         safe_auc(mink_a, labs),
        "mink_member":     float(mem.mean()) if len(mem) else 0.0,
        "mink_nonmember":  float(non.mean()) if len(non) else 0.0,
        "mink_gap":        float(mem.mean() - non.mean()) if len(mem) and len(non) else 0.0,
        "ppl_auc":         safe_auc(ppl_s, labs),
        "lowercase_auc":   safe_auc(lower_s, labs),
        "zlib_auc":        safe_auc(zlib_s, labs),
        "smaller_ref_auc": safe_auc(ref_s, labs) if ref_model is not None else 0.5,
        "ppl_member":      float(ppl_mem.mean()) if len(ppl_mem) else 0.0,
        "ppl_nonmember":   float(ppl_non.mean()) if len(ppl_non) else 0.0,
    }


# ── Verbatim / extraction memorization (direct, generation-based) ─────────────

def verbatim_memorization_rate(
    model,
    tokenizer,
    texts: List[str],
    prefix_tokens: int = 32,
    continuation_tokens: int = 32,
    max_length=MAX_LENGTH,
) -> Dict[str, float]:
    """Greedy-decode from each text's prefix and measure how much of the true
    continuation the model reproduces verbatim.

    Returns:
      exact_match_rate: fraction of examples whose greedy continuation exactly
                        matches the reference continuation token-for-token.
      token_overlap:    mean fraction of matching continuation tokens (position-wise).
    Both drop as the model de-memorizes. Only texts long enough for
    prefix+continuation are counted.
    """
    model.eval()
    exact, overlaps, counted = 0, [], 0

    with torch.no_grad():
        for text in texts:
            ids = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=max_length)["input_ids"][0]
            if ids.numel() < prefix_tokens + continuation_tokens:
                continue
            counted += 1
            prefix = ids[:prefix_tokens].unsqueeze(0).to(model.device)
            true_cont = ids[prefix_tokens:prefix_tokens + continuation_tokens]
            gen = model.generate(
                prefix,
                max_new_tokens=continuation_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )[0]
            pred_cont = gen[prefix_tokens:prefix_tokens + continuation_tokens].cpu()
            m = min(pred_cont.numel(), true_cont.numel())
            match = (pred_cont[:m] == true_cont[:m]).float().mean().item() if m else 0.0
            overlaps.append(match)
            if m == continuation_tokens and match == 1.0:
                exact += 1

    return {
        "exact_match_rate": exact / counted if counted else 0.0,
        "token_overlap":    float(np.mean(overlaps)) if overlaps else 0.0,
        "num_evaluated":    counted,
    }


# ── Before/after change ───────────────────────────────────────────────────────

def memorization_change(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Diff two memorization panels (after − before) with a sign convention.

    For every metric, `improved` is True when the change indicates LESS
    memorization / leakage:
      - AUCs and mink_gap  -> moved toward chance (|value − 0.5| or |gap| shrank)
      - ppl_member         -> increased (model less confident on members)
    """
    out: Dict[str, Dict[str, float]] = {}
    for key in before:
        if key not in after:
            continue
        b, a = before[key], after[key]
        delta = a - b
        if key in _AUC_KEYS:
            improved = abs(a - 0.5) < abs(b - 0.5)
        elif key == "mink_gap":
            improved = abs(a) < abs(b)
        elif key == "ppl_member":
            improved = a > b
        else:
            improved = None  # descriptive stat, no direction
        out[key] = {"before": b, "after": a, "delta": delta, "improved": improved}
    return out


def analyze_unlearning(
    model,
    tokenizer,
    dataset,
    apply_method: Callable[[], None],
    ref_model=None,
    ref_tokenizer=None,
    forget_texts: Optional[List[str]] = None,
    verbatim: bool = False,
    k=MIN_K_FRAC,
) -> Dict[str, object]:
    """Snapshot memorization, apply a de-memorization method, snapshot again.

    apply_method: zero-arg callable that mutates `model` in place (e.g. a
                  lambda wrapping DELUnlearning(model).unlearn(...) or a pruning
                  call). Runs between the two snapshots.
    forget_texts: member texts for the optional verbatim/extraction metric.

    Returns {"before", "after", "change"} and, if verbatim=True,
    {"verbatim_before", "verbatim_after"}.
    """
    before = evaluate_memorization(model, tokenizer, dataset, ref_model,
                                   ref_tokenizer, k=k, desc="before")
    vb_before = (verbatim_memorization_rate(model, tokenizer, forget_texts)
                 if verbatim and forget_texts else None)

    apply_method()

    after = evaluate_memorization(model, tokenizer, dataset, ref_model,
                                  ref_tokenizer, k=k, desc="after")
    vb_after = (verbatim_memorization_rate(model, tokenizer, forget_texts)
                if verbatim and forget_texts else None)

    result: Dict[str, object] = {
        "before": before,
        "after": after,
        "change": memorization_change(before, after),
    }
    if vb_before is not None:
        result["verbatim_before"] = vb_before
        result["verbatim_after"] = vb_after
    return result


def print_change(change: Dict[str, Dict[str, float]], title: str = "Memorization change"):
    """Pretty-print the output of `memorization_change`."""
    print(f"\n{title}")
    print(f"{'metric':<18}{'before':>10}{'after':>10}{'delta':>10}  {'':<4}")
    print("-" * 56)
    for key, d in change.items():
        flag = "" if d["improved"] is None else ("↓mem" if d["improved"] else "↑mem")
        print(f"{key:<18}{d['before']:>10.4f}{d['after']:>10.4f}{d['delta']:>+10.4f}  {flag:<4}")
