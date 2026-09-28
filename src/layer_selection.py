"""Architecture-aware regions, reversible interventions, and activation rank."""
from contextlib import contextmanager
import random
import re


def layer_index(name):
    match = re.search(r"(?:^|\.)(?:layers|h)\.(\d+)(?:\.|$)", name)
    return int(match.group(1)) if match else None


def transformer_blocks(model):
    for path in ("gpt_neox.layers", "model.layers", "transformer.h"):
        node = model
        for part in path.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        if node is not None:
            return list(node)
    raise ValueError(f"Unsupported transformer architecture: {type(model).__name__}")


def attention_parameters(model, layers=None):
    selected = None if layers is None else set(layers)
    result = {}
    for name, parameter in model.named_parameters():
        index = layer_index(name)
        if index is None or (selected is not None and index not in selected):
            continue
        if name.endswith(tuple(f"{projection}.weight" for projection in
                               ("query_key_value", "q_proj", "k_proj", "v_proj", "c_attn"))):
            result[name] = parameter
    if not result:
        raise ValueError("No attention Q/K/V matrices matched the selected region")
    return result


def choose_layers(n_layers, count, strategy, scores=None, seed=42, windows=None):
    if not 1 <= count <= n_layers:
        raise ValueError("Selected layer count must be between 1 and model depth")
    if strategy == "early":
        return list(range(count))
    if strategy == "middle":
        start = (n_layers - count) // 2
        return list(range(start, start + count))
    if strategy == "late":
        return list(range(n_layers - count, n_layers))
    if strategy == "random":
        return sorted(random.Random(seed).sample(range(n_layers), count))
    if strategy.endswith("_window"):
        candidates = [r for r in (windows or []) if len(r["layers"]) == count]
        if not candidates:
            raise ValueError("No measured windows match the requested layer count")
        return min(candidates, key=lambda r: (-r["score"], r["layers"]))["layers"]
    if scores is None or set(scores) != set(range(n_layers)):
        raise ValueError("A score for every layer is required")
    return sorted(sorted(scores, key=lambda i: (-scores[i], i))[:count])


def scan_regions(n_layers, widths):
    regions = [[i] for i in range(n_layers)]
    for width in sorted(set(widths)):
        if not 1 <= width <= n_layers:
            raise ValueError("Window width outside model depth")
        if width > 1:
            regions.extend([list(range(start, start + width)) for start in range(n_layers - width + 1)])
    return regions


def prune_attention(model, layers, ratio):
    import torch
    if not 0 <= ratio <= 1:
        raise ValueError("Pruning ratio must be between zero and one")
    masks = {}
    with torch.no_grad():
        for name, parameter in attention_parameters(model, layers).items():
            mask = torch.zeros(parameter.numel(), dtype=torch.bool, device=parameter.device)
            count = round(parameter.numel() * ratio)
            if count:
                ids = parameter.detach().abs().flatten().topk(count, largest=False).indices
                mask[ids] = True
            mask = mask.reshape(parameter.shape)
            parameter.masked_fill_(mask, 0)
            masks[name] = mask
    return masks


@contextmanager
def intervention(model, layers, kind="pruning", ratio=0.1):
    """Restore parameters/hooks even if evaluation fails.

    FIT-style proxy: replace each selected block's residual output with its
    input and measure the loss change. This is a loss-sensitivity ablation,
    not a claim to reproduce a specific FIT implementation.
    """
    import torch
    handles, backups = [], {}
    try:
        if kind == "pruning":
            params = attention_parameters(model, layers)
            backups = {n: p.detach().cpu().clone() for n, p in params.items()}
            prune_attention(model, layers, ratio)
        elif kind == "masking_loss":
            def bypass(module, args, kwargs, output):
                hidden = args[0] if args else kwargs["hidden_states"]
                return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
            for i in layers:
                handles.append(transformer_blocks(model)[i].register_forward_hook(bypass, with_kwargs=True))
        else:
            raise ValueError(f"Unknown intervention: {kind}")
        yield
    finally:
        for handle in handles:
            handle.remove()
        with torch.no_grad():
            params = dict(model.named_parameters())
            for name, values in backups.items():
                params[name].copy_(values.to(params[name].device))


def effective_rank(matrix):
    """exp(entropy(normalized singular values)); zero matrix has rank zero."""
    import torch
    values = torch.linalg.svdvals(matrix.float())
    values = values[values > 0]
    if not len(values):
        return 0.0
    probabilities = values / values.sum()
    return float(torch.exp(-(probabilities * probabilities.log()).sum()))


def activation_ranks(model, tokenizer, rows, max_length=512, max_tokens=128):
    """Rank of uncentered block-output activations, fixed first-N token sample."""
    import torch
    if max_tokens < 1:
        raise ValueError("Rank token budget must be positive")
    blocks = transformer_blocks(model)
    samples, counts, handles = {i: [] for i in range(len(blocks))}, {}, []
    def hook(index):
        def collect(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            left = max_tokens - counts.get(index, 0)
            if left > 0:
                values = hidden.detach().reshape(-1, hidden.shape[-1])[:left].float().cpu()
                samples[index].append(values)
                counts[index] = counts.get(index, 0) + len(values)
        return collect
    was_training = model.training
    model.eval()
    try:
        for i, block in enumerate(blocks):
            handles.append(block.register_forward_hook(hook(i)))
        with torch.no_grad():
            for row in rows:
                batch = tokenizer(row["input"], return_tensors="pt", truncation=True,
                                  max_length=max_length, add_special_tokens=False)
                model(**{k: v.to(next(model.parameters()).device) for k, v in batch.items()})
                if all(counts.get(i, 0) >= max_tokens for i in samples):
                    break
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return {str(i): {"effective_rank": effective_rank(torch.cat(values)),
                     "n_tokens": counts[i], "definition": "uncentered block-output activation singular-value entropy"}
            for i, values in samples.items() if values}


def parameter_record(model, names=None):
    selected = None if names is None else set(names)
    result = {}
    for name, parameter in model.named_parameters():
        if selected is not None and name not in selected:
            continue
        result[name] = {"shape": list(parameter.shape), "numel": parameter.numel(),
                        "zeros": int((parameter.detach() == 0).sum()),
                        "requires_grad": parameter.requires_grad}
    return result
