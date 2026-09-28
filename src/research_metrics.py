"""Attack, extraction, and token-weighted LM metrics with exact denominators."""
import math
from pathlib import Path

import numpy as np

from experiment_protocol import digest, text_id, write_json


def roc_points(labels, scores):
    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    if len(labels) != len(scores) or not len(labels) or not np.isfinite(scores).all():
        raise ValueError("Scores must be aligned, nonempty, and finite")
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("ROC requires both membership classes")
    order = np.argsort(-scores, kind="stable")
    y, s = labels[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(y)[ends]
    fp = 1 + ends - tp
    return np.r_[0., fp / (len(y) - y.sum())], np.r_[0., tp / y.sum()]


def auc(labels, scores):
    fpr, tpr = roc_points(labels, scores)
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


def attack_metrics(labels, scores, *, direction=1, bootstrap=1000, seed=42,
                   low_fprs=(0.001, 0.01)):
    """Direction must be frozen on tuning data. Oracle max-AUC is descriptive.

    Advantage = max_threshold(TPR - FPR) for the frozen score orientation.
    Bootstrap resamples within each class, preserving class counts.
    """
    if direction not in (-1, 1) or bootstrap < 0:
        raise ValueError("Invalid orientation or bootstrap count")
    y, s = np.asarray(labels), np.asarray(scores, dtype=float)
    raw = auc(y, s)
    fpr, tpr = roc_points(y, direction * s)
    adjusted = auc(y, direction * s)
    result = {"raw_auc": raw, "oriented_auc": adjusted, "direction": direction,
              "max_auc": max(raw, 1 - raw), "advantage": float(np.max(tpr - fpr)),
              "advantage_definition": "max_threshold(TPR-FPR), tuning-frozen orientation",
              "n": len(y), "n_members": int(y.sum()), "n_nonmembers": int((y == 0).sum()),
              "low_fpr": {str(v): {"tpr": float(tpr[fpr <= v].max()),
                                    "allowed_false_positives": int(v * (y == 0).sum())}
                          for v in low_fprs},
              "bootstrap_replicates": bootstrap, "bootstrap_seed": seed}
    if bootstrap:
        rng = np.random.default_rng(seed)
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        samples = []
        for _ in range(bootstrap):
            ids = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
            value = auc(y[ids], s[ids])
            samples.append((value, value if direction == 1 else 1 - value, max(value, 1 - value)))
        intervals = np.quantile(samples, [0.025, 0.975], axis=0)
        result["ci95"] = {name: intervals[:, i].tolist()
                          for i, name in enumerate(("raw_auc", "oriented_auc", "max_auc"))}
    return result


def longest_matching_span(a, b):
    """Longest contiguous common token span, allowing different start positions."""
    previous, best = [0] * (len(b) + 1), 0
    for token in a:
        current = [0] * (len(b) + 1)
        for j, other in enumerate(b, 1):
            if token == other:
                current[j] = previous[j - 1] + 1
                best = max(best, current[j])
        previous = current
    return best


def extraction_metrics(target, generated):
    if not target:
        raise ValueError("Empty continuation")
    return {"exact_match": int(list(target) == list(generated)),
            "recovered_tokens": sum(a == b for a, b in zip(target, generated)),
            "target_tokens": len(target),
            "token_recovery": sum(a == b for a, b in zip(target, generated)) / len(target),
            "longest_matching_span": longest_matching_span(target, generated)}


def wilson_interval(successes, total):
    if not total:
        return None
    z = 1.959963984540054
    p, denom = successes / total, 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [max(0., center - half), min(1., center + half)]


def context_limit(model, requested):
    limit = getattr(model.config, "max_position_embeddings", requested)
    return min(requested, limit)


def score_examples(model, tokenizer, rows, max_length=512, prefix_length=32,
                   continuation_length=32, min_k=0.2):
    import torch
    import torch.nn.functional as F
    if prefix_length < 1 or continuation_length < 1 or not 0 < min_k <= 1:
        raise ValueError("Invalid extraction lengths or Min-K fraction")
    if prefix_length + continuation_length > context_limit(model, max_length):
        raise ValueError("Extraction lengths exceed the configured/model context")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    results = []
    try:
        with torch.no_grad():
            for row in rows:
                ids = tokenizer(row["input"], add_special_tokens=False)["input_ids"]
                ids = ids[:context_limit(model, max_length)]
                result = {"id": row["id"], "label": row.get("label"), "tokens": len(ids)}
                if len(ids) < 2:
                    results.append(dict(result, status="too_short"))
                    continue
                x = torch.tensor([ids], device=device)
                logits = model(input_ids=x).logits[:, :-1].float()
                logp = F.log_softmax(logits, -1).gather(-1, x[:, 1:, None]).flatten()
                if not torch.isfinite(logp).all():
                    raise FloatingPointError(f"Nonfinite log probabilities: {row['id']}")
                result.update(status="ok", nll=float(-logp.mean()),
                              min_k=float(logp.topk(max(1, int(len(logp) * min_k)), largest=False).values.mean()))
                if len(ids) >= prefix_length + continuation_length:
                    prefix = x[:, :prefix_length]
                    generated = model.generate(input_ids=prefix, attention_mask=torch.ones_like(prefix),
                                               max_new_tokens=continuation_length, do_sample=False,
                                               num_beams=1, pad_token_id=tokenizer.pad_token_id,
                                               use_cache=True)[0, prefix_length:].tolist()
                    target = ids[prefix_length:prefix_length + continuation_length]
                    result["extraction"] = dict(extraction_metrics(target, generated),
                                                prefix_tokens=ids[:prefix_length],
                                                target=target, generated=generated)
                else:
                    result["extraction_skip"] = "insufficient_tokens"
                results.append(result)
    finally:
        model.train(was_training)
    return results


def summarize_scores(rows, *, direction=1, bootstrap=1000, seed=42):
    valid = [r for r in rows if r["status"] == "ok"]
    extra = [r["extraction"] for r in valid if "extraction" in r]
    result = {"n_requested": len(rows), "n_scored": len(valid),
              "n_skipped": len(rows) - len(valid), "extraction_n": len(extra)}
    if valid:
        tokens = sum(r["tokens"] - 1 for r in valid)
        result["mean_nll"] = sum(r["nll"] * (r["tokens"] - 1) for r in valid) / tokens
        result["n_loss_tokens"] = tokens
    if {r["label"] for r in valid} == {0, 1}:
        result["attack"] = attack_metrics([r["label"] for r in valid], [r["min_k"] for r in valid],
                                          direction=direction, bootstrap=bootstrap, seed=seed)
    else:
        result["attack_unavailable"] = "both membership classes required"
    if extra:
        n = len(extra)
        exact = sum(r["exact_match"] for r in extra)
        result["extraction"] = {"exact_match": exact / n,
                                "exact_match_ci95": wilson_interval(exact, n),
                                "token_recovery": sum(r["recovered_tokens"] for r in extra) / sum(r["target_tokens"] for r in extra),
                                "longest_matching_span": sum(r["longest_matching_span"] for r in extra) / n}
    return result


def build_utility_corpora(tokenizer, texts, excluded_rows, *, seed=42, token_budget=131072, block_size=512):
    """Disjoint document partitions; fixed overlapping-by-one blocks per role.

    Every token except the first token in a partition is a prediction target.
    Concatenation uses EOS boundaries. Short final blocks remain included.
    """
    import random
    if token_budget < 2 or block_size < 2:
        raise ValueError("Utility token budget and block size must be >= 2")
    excluded = {text_id(r["input"]) for r in excluded_rows}
    documents = {text_id(t): t for t in texts if t.strip() and text_id(t) not in excluded}
    items = sorted(documents.items())
    random.Random(seed).shuffle(items)
    corpora = {}
    for index, role in enumerate(("selection", "tuning", "evaluation")):
        stream, used = [], []
        for doc_id, text in items[index::3]:
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if tokenizer.eos_token_id is not None:
                ids.append(tokenizer.eos_token_id)
            if not ids:
                continue
            take = min(len(ids), token_budget - len(stream))
            stream.extend(ids[:take])
            used.append({"id": doc_id, "tokens_used": take})
            if len(stream) >= token_budget:
                break
        if len(stream) < 2:
            raise ValueError(f"Insufficient utility data for {role}")
        blocks = [stream[i:i + block_size] for i in range(0, len(stream) - 1, block_size - 1)]
        corpora[role] = {"blocks": blocks, "documents": used, "requested_tokens": token_budget,
                         "actual_tokens": len(stream), "token_hash": digest(stream),
                         "tokenizer": tokenizer.name_or_path, "block_size": block_size}
    return corpora


def corpus_loss(model, corpus):
    import torch
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    nll, count = 0., 0
    try:
        with torch.no_grad():
            for ids in corpus["blocks"]:
                if len(ids) > context_limit(model, len(ids)):
                    raise ValueError("Utility block exceeds model context")
                x = torch.tensor([ids], device=device)
                loss = float(model(input_ids=x, labels=x).loss)
                if not math.isfinite(loss):
                    raise FloatingPointError("Nonfinite utility loss; run is invalid")
                nll += loss * (len(ids) - 1)
                count += len(ids) - 1
    finally:
        model.train(was_training)
    if not count:
        raise ValueError("No utility prediction tokens")
    mean = nll / count
    return {"nll": mean, "perplexity": math.exp(mean) if mean < 709 else None,
            "perplexity_overflow": mean >= 709, "n_tokens": count,
            "n_blocks": len(corpus["blocks"]), "token_hash": corpus["token_hash"]}


def plot_scores(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for label, name in ((0, "Non-members"), (1, "Members")):
        values = [r["min_k"] for r in rows if r["status"] == "ok" and r["label"] == label]
        if values:
            ax.hist(values, bins=30, alpha=0.5, label=f"{name} (n={len(values)})", density=True)
    ax.set(xlabel="Min-K% score (raw)", ylabel="Density")
    ax.legend()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
