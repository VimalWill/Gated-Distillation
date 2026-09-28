"""Auditable experiment runner. See EXPERIMENTS.md for protocols and examples."""
import argparse
from copy import deepcopy
import itertools
import math
from pathlib import Path
import random
import sys

from experiment_protocol import (ROLES, assert_disjoint, canonical_records, digest,
                                 make_splits, provenance, read_json, records_from_jsonl,
                                 seed_everything, validate_splits, write_json)


def load_model(config, source=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    source = source or config["model"]
    kwargs = {} if Path(source).exists() else {"revision": config.get("revision", "main")}
    tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = getattr(torch, config.get("dtype", "float32"))
    model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=dtype, **kwargs)
    model.to(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    return model, tokenizer


def release(*models):
    # Callers discard references before invoking this; CUDA cache cleanup is optional.
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def metric_args(config):
    return {k: config.get(k, v) for k, v in
            (("max_length", 512), ("prefix_length", 32), ("continuation_length", 32), ("min_k", 0.2))}


def evaluate(model, tokenizer, rows, corpus, config, directory, direction=1, bootstrap=None):
    from research_metrics import corpus_loss, plot_scores, score_examples, summarize_scores
    scored = score_examples(model, tokenizer, rows, **metric_args(config))
    settings = dict(direction=direction, seed=config["seed"],
                    bootstrap=config.get("bootstrap", 1000) if bootstrap is None else bootstrap)
    summary = summarize_scores(scored, **settings)
    summary["by_class"] = {str(label): summarize_scores([r for r in scored if r["label"] == label],
                                                       bootstrap=0)
                           for label in (0, 1)}
    summary["utility"] = corpus_loss(model, corpus)
    if directory is not None:
        write_json(Path(directory) / "scores.json", scored)
        write_json(Path(directory) / "metrics.json", summary)
        if config.get("plots", True):
            plot_scores(scored, Path(directory) / "scores.png")
    return summary, scored


def utility_data(config, tokenizer, bundle):
    from research_metrics import build_utility_corpora
    source = config.get("utility_jsonl")
    if source:
        texts = [r.get("input", r.get("text")) for r in records_from_jsonl(source)]
    else:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test",
                               revision=config.get("utility_revision", "main"))
        texts = dataset["text"]
    excluded = [r for rows in bundle["roles"].values() for r in rows]
    # Also exclude optional durability training and neighbor evaluation documents.
    for key in ("benign_jsonl", "neighbors_jsonl"):
        if config.get(key):
            excluded.extend(canonical_records(records_from_jsonl(config[key]), require_labels=False))
    return build_utility_corpora(tokenizer, texts, excluded, seed=config["split_seed"],
                                token_budget=config.get("utility_tokens", 131072),
                                block_size=config.get("max_length", 512))


def prepare(args):
    if args.input:
        rows = records_from_jsonl(args.input)
        source = {"path": str(Path(args.input).resolve())}
    else:
        from datasets import load_dataset
        rows = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{args.length}", revision=args.revision)
        source = {"dataset": "swj0419/WikiMIA", "split": f"WikiMIA_length{args.length}", "revision": args.revision}
    bundle = make_splits(rows, args.seed, tuple(args.fractions))
    bundle["source"] = source
    if Path(args.output).exists():
        if read_json(args.output) != bundle:
            raise FileExistsError("Existing split bundle differs; choose a new path")
    else:
        write_json(args.output, bundle)
    print(f"Saved {args.output}: " + ", ".join(f"{role}={len(rows)}" for role, rows in bundle["roles"].items()))


def supervised_training(model, tokenizer, rows, config):
    """Benign/controlled/relearning training; log actual example and token exposures."""
    import torch
    seed_everything(config["seed"])
    if not rows or config.get("epochs", 1) < 1:
        raise ValueError("Supervised training requires data and positive epochs")
    for parameter in model.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 1e-5), weight_decay=0.)
    exposures, steps = {}, 0
    model.train()
    for epoch in range(config.get("epochs", 1)):
        order = list(rows)
        random.Random(config["seed"] + epoch).shuffle(order)
        for row in order:
            ids = tokenizer(row["input"], add_special_tokens=False)["input_ids"][:config.get("max_length", 512)]
            if len(ids) < 2:
                continue
            batch = torch.tensor([ids], device=next(model.parameters()).device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=batch, labels=batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss: {row['id']}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            log = exposures.setdefault(row["id"], {"presentations": 0, "tokens_per_presentation": len(ids),
                                                    "token_hash": digest(ids)})
            log["presentations"] += 1
            steps += 1
            if config.get("max_steps") is not None and steps >= config["max_steps"]:
                break
        if config.get("max_steps") is not None and steps >= config["max_steps"]:
            break
    model.eval()
    return {"optimizer_steps": steps, "exposures": exposures,
            "loss_tokens": sum((r["tokens_per_presentation"] - 1) * r["presentations"] for r in exposures.values())}


def controlled(args):
    """Train known-exposure checkpoints, one fresh initialization per duplication level."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    config = read_json(args.config)
    bundle = validate_splits(read_json(config["splits"]))
    rows = [r for role in ROLES for r in bundle["roles"][role]]
    members = [r for r in rows if r["label"] == 1]
    selected = set(config.get("insert_ids", [r["id"] for r in members]))
    if not selected <= {r["id"] for r in members}:
        raise ValueError("insert_ids must refer only to labeled members")
    if config.get("max_steps") is not None:
        raise ValueError("Controlled training cannot truncate exposure using max_steps")
    background = canonical_records(records_from_jsonl(config["background_jsonl"]), False) if config.get("background_jsonl") else []
    assert_disjoint({"membership": rows, "background": background})
    for copies in config.get("duplication_levels", [1, 5, 10]):
        if not isinstance(copies, int) or copies < 1:
            raise ValueError("Duplication levels must be positive integers")
        for seed in config.get("seeds", [42]):
            out = Path(config["output"]) / f"copies-{copies}" / f"seed-{seed}"
            if out.exists():
                raise FileExistsError(out)
            seed_everything(seed)
            init = config.get("initialization", "scratch")
            if init == "scratch":
                tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config.get("revision", "main"))
                architecture = AutoConfig.from_pretrained(config["model"], revision=config.get("revision", "main"))
                for key, value in config.get("architecture_overrides", {}).items():
                    setattr(architecture, key, value)
                model = AutoModelForCausalLM.from_config(architecture)
                model.to(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
            elif init == "pretrained":
                model, tokenizer = load_model(config)
            else:
                raise ValueError("initialization must be scratch or pretrained")
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            training_rows = background + [r for r in members for _ in range(copies if r["id"] in selected else 1)]
            log = supervised_training(model, tokenizer, training_rows, dict(config, seed=seed))
            missing = {r["id"] for r in members} - set(log["exposures"])
            if missing:
                raise ValueError(f"Members received no training exposure: {sorted(missing)}")
            model.save_pretrained(out)
            tokenizer.save_pretrained(out)
            manifest = provenance(dict(config, seed=seed, copies=copies), model)
            manifest.update(dataset_hash=bundle["dataset_hash"],
                            membership_basis="known_from_scratch" if init == "scratch" else "known_added_exposure_only_pretraining_unknown",
                            nonmember_ids=[r["id"] for r in rows if r["label"] == 0],
                            training=log, checkpoint=str(out.resolve()))
            write_json(out / "membership_manifest.json", manifest)
            print(f"Controlled checkpoint: {out}")
            del model
            release()


def profile(model, tokenizer, bundle, corpora, config, out):
    from layer_selection import activation_ranks, intervention, scan_regions, transformer_blocks
    selection = bundle["roles"]["selection"]
    baseline, _ = evaluate(model, tokenizer, selection, corpora["selection"], config, out / "baseline", bootstrap=0)
    ranks = activation_ranks(model, tokenizer, selection, config.get("max_length", 512), config.get("rank_tokens", 128))
    records = []
    for kind in ("pruning", "masking_loss"):
        for layers in scan_regions(len(transformer_blocks(model)), config.get("window_widths", [])):
            with intervention(model, layers, kind, config.get("prune_ratio", 0.1)):
                metrics, _ = evaluate(model, tokenizer, selection, corpora["selection"], config, None, bootstrap=0)
            utility_delta = metrics["utility"]["nll"] - baseline["utility"]["nll"]
            member_delta = metrics["by_class"]["1"]["mean_nll"] - baseline["by_class"]["1"]["mean_nll"]
            auc_reduction = baseline["attack"]["max_auc"] - metrics["attack"]["max_auc"]
            extraction_reduction = None
            if baseline["by_class"]["1"].get("extraction") and metrics["by_class"]["1"].get("extraction"):
                extraction_reduction = (baseline["by_class"]["1"]["extraction"]["token_recovery"] -
                                        metrics["by_class"]["1"]["extraction"]["token_recovery"])
            score_name = config.get("pruning_score", "auc_reduction") if kind == "pruning" else "member_nll_increase"
            options = {"auc_reduction": auc_reduction, "member_nll_increase": member_delta,
                       "extraction_reduction": extraction_reduction}
            if score_name not in options or options[score_name] is None:
                raise ValueError(f"Selection score unavailable: {score_name}; check extraction lengths")
            record = {"kind": kind, "layers": layers, "metrics": metrics,
                      "utility_nll_delta": utility_delta, "member_nll_delta": member_delta,
                      "auc_reduction": auc_reduction, "extraction_reduction": extraction_reduction,
                      "score_name": score_name, "score": options[score_name],
                      "utility_normalized_score": options[score_name] / utility_delta if utility_delta > 1e-8 else None,
                      "activation_ranks": {str(i): ranks[str(i)] for i in layers}}
            records.append(record)
            write_json(out / "profile.json", {"baseline": baseline, "ranks": ranks, "interventions": records,
                                             "config": config, "selection_ids": [r["id"] for r in selection]})
            print(f"Profile {kind}, layers={layers}: score={record['score']:.5f}, utility ΔNLL={utility_delta:.5f}", flush=True)
    return records, ranks


def setup_student(model, layers, config):
    from layer_selection import attention_parameters, layer_index, prune_attention, transformer_blocks
    # Same global pruning intervention for every selector within a matched group.
    masks = prune_attention(model, range(len(transformer_blocks(model))), config.get("prune_ratio", 0.1))
    variant = config.get("variant", "STUDE")
    if variant == "LoRA-STUDE":
        from peft import LoraConfig, get_peft_model
        names = attention_parameters(model, layers)
        suffixes = sorted({name.split(".")[-2] for name in names})
        pattern = "h" if any("transformer.h." in n for n in names) else "layers"
        model = get_peft_model(model, LoraConfig(r=config.get("lora_rank", 8),
                                               lora_alpha=config.get("lora_alpha", 16),
                                               lora_dropout=0., target_modules=suffixes,
                                               layers_to_transform=layers, layers_pattern=pattern,
                                               bias="none", task_type="CAUSAL_LM"))
    elif variant == "STUDE":
        targets = set(attention_parameters(model, layers))
        for name, parameter in model.named_parameters():
            parameter.requires_grad = (name in targets if config.get("update_scope", "block") == "attention"
                                       else layer_index(name) in layers)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not count:
        raise ValueError("Selected region has no trainable parameters")
    return model, masks, count


def unlearn_epoch(model, reference, tokenizer, rows, config, masks, epoch):
    import torch
    import torch.nn.functional as F
    # An optimizer is passed through config-independent state by the caller.
    optimizer = config["optimizer"]
    accumulation = config.get("gradient_accumulation", 8)
    if accumulation < 1:
        raise ValueError("gradient_accumulation must be positive")
    order = list(rows)
    random.Random(config["seed"] + epoch).shuffle(order)
    tokenized = [(r["id"], tokenizer(r["input"], add_special_tokens=False)["input_ids"][:config.get("max_length", 512)]) for r in order]
    if any(len(ids) < 2 for _, ids in tokenized):
        raise ValueError("Forget examples must have at least two tokens")
    if config.get("max_examples") is not None:
        tokenized = tokenized[:config["max_examples"]]
    model.train()
    reference.eval()
    count = 0
    for start in range(0, len(tokenized), accumulation):
        group = tokenized[start:start + accumulation]
        optimizer.zero_grad(set_to_none=True)
        for _, ids in group:
            x = torch.tensor([ids], device=next(model.parameters()).device)
            logp = F.log_softmax(model(input_ids=x).logits[:, :-1].float(), -1)
            actual = logp.gather(-1, x[:, 1:, None]).flatten()
            k = max(1, int(len(actual) * config.get("objective_min_k", 0.)))
            # Minimize log-probability of the least likely tokens (ascent on NLL).
            ascent = actual.topk(k, largest=False).values.mean()
            with torch.no_grad():
                ref_logp = F.log_softmax(reference(input_ids=x).logits[:, :-1].float(), -1)
            kl = F.kl_div(logp, ref_logp, log_target=True, reduction="sum") / actual.numel()
            loss = (ascent + config.get("kl_weight", 0.1) * kl) / len(group)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite unlearning loss")
            loss.backward()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in masks and parameter.grad is not None:
                    parameter.grad.masked_fill_(masks[name], 0)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        optimizer.step()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in masks:
                    parameter.masked_fill_(masks[name], 0)
        count += 1
    model.eval()
    return {"optimizer_steps": count, "example_ids": [i for i, _ in tokenized],
            "loss_tokens": sum(len(ids) - 1 for _, ids in tokenized)}


def frozen_direction(scored):
    from research_metrics import auc
    valid = [r for r in scored if r["status"] == "ok"]
    return 1 if auc([r["label"] for r in valid], [r["min_k"] for r in valid]) >= 0.5 else -1


def save_student(model, tokenizer, out, config):
    # Keep adapters intact for continued epoch training; merge only after selection.
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)


def downstream(model, tokenizer, config, out):
    if not config.get("downstream", False):
        return {"status": "not_requested"}
    from lm_eval_utils import run_lm_eval
    scores = run_lm_eval(model, tokenizer, tasks=config.get("downstream_tasks"),
                         limit=config.get("downstream_limit"),
                         batch_size=config.get("downstream_batch_size", 4),
                         details_path=out / "downstream_details.json", seed=config["seed"])
    return {"status": "complete" if scores else "unavailable_or_failed", "accuracy": scores}


def causal_loader(rows, tokenizer, config):
    import torch
    from torch.utils.data import DataLoader
    def collate(batch):
        enc = tokenizer([r["input"] for r in batch], return_tensors="pt", padding=True,
                        truncation=True, max_length=config.get("max_length", 512))
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        enc["labels"] = labels
        return enc
    generator = torch.Generator().manual_seed(config["seed"])
    return DataLoader(rows, batch_size=config.get("baseline_batch_size", 4),
                      shuffle=True, generator=generator, collate_fn=collate)


def run_baselines(config, bundle, corpora, baseline_tuning, out):
    """Whole-method comparisons have distinct objectives; selectors are matched separately."""
    from layer_selection import attention_parameters, parameter_record
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    results = {}
    forget = [r for r in bundle["roles"]["optimization"] if r["label"] == 1]
    retain = [r for r in bundle["roles"]["optimization"] if r["label"] == 0]
    for name in config.get("baselines", ["del", "spe"]):
        defaults = ([{"budget_alpha": 0.005, "lr": 5e-5, "epochs": 3}] if name == "del"
                    else [{"sparsity": 0.9, "lr": 1e-7}])
        grid = config.get("baseline_grids", {}).get(name, defaults)
        directory, best, trials = out / name, None, []
        for knobs in grid:
            seed_everything(config["seed"])
            model, tokenizer = load_model(config)
            targets = list(attention_parameters(model))
            forget_loader, retain_loader = (causal_loader(rows, tokenizer, config) for rows in (forget, retain))
            if name == "del":
                from methods.del_unlearning import DELUnlearning
                method = DELUnlearning(model)
                criticality = method.compute_criticality_scores(forget_loader, targets, top_h=knobs.get("top_h", 5))
                mask = method.generate_mask(criticality, budget_alpha=knobs["budget_alpha"])
                if not any(bool(m.any()) for m in mask.values()):
                    raise ValueError("DEL budget selects no parameters; increase budget_alpha")
                method.reset_parameters(mask)
                method.finetune_masked_params(retain_loader, mask, learning_rate=knobs["lr"], epochs=knobs.get("epochs", 1))
            elif name == "spe":
                from methods.spe_unlearning import SPEUnlearning
                method = SPEUnlearning(model)
                mask = method.unlearn(retain_loader, forget_loader, layer_names=targets,
                                      sparsity=knobs["sparsity"], learning_rate=knobs["lr"],
                                      damping=knobs.get("damping", 0.01), max_update=knobs.get("max_update", 1.))
            else:
                raise ValueError(f"Unsupported whole-method baseline: {name}")
            metrics, scored = evaluate(model, tokenizer, bundle["roles"]["tuning"], corpora["tuning"], config, None, bootstrap=0)
            in_budget = metrics["utility"]["nll"] <= baseline_tuning["utility"]["nll"] + math.log(config.get("utility_budget_mult", 1.5))
            gap = abs(metrics["attack"]["raw_auc"] - 0.5)
            trial = {"knobs": knobs, "metrics": metrics, "in_budget": in_budget, "gap": gap,
                     "update_mask_counts": {n: int(m.sum()) for n, m in mask.items()},
                     "updated_parameter_budget": sum(int(m.sum()) for m in mask.values())}
            trials.append(trial)
            if in_budget and (best is None or gap < best["gap"]):
                best = dict(trial, direction=frozen_direction(scored))
                model.save_pretrained(directory / "checkpoint")
                tokenizer.save_pretrained(directory / "checkpoint")
            write_json(directory / "tuning.json", {"trials": trials, "best": best})
            del method, model, mask
            release()
        if best is None:
            results[name] = {"status": "no_checkpoint_within_utility_budget"}
            continue
        model, tokenizer = load_model(config, str(directory / "checkpoint"))
        result = {"status": "complete", "comparison_type": "whole_method_distinct_objective"}
        for role in ("optimization", "evaluation"):
            result[role], _ = evaluate(model, tokenizer, bundle["roles"][role], corpora["evaluation"],
                                       config, directory / role, direction=best["direction"])
        result["downstream"] = downstream(model, tokenizer, config, directory)
        record = provenance(dict(config, method=name), model)
        record.update(best_trial=best, split_hash=digest(bundle), tuning_candidates=len(grid),
                      deployment_attention=parameter_record(model, attention_parameters(model)),
                      optimization_ids={"forget": [r["id"] for r in forget], "retain": [r["id"] for r in retain]},
                      objective="DEL reset+retain fine-tuning" if name == "del" else "SPE Fisher/forget-gradient update")
        write_json(directory / "manifest.json", record)
        results[name] = result
        del model
        release()
    return results


def load_student(path, config):
    if config.get("variant", "STUDE") != "LoRA-STUDE":
        return load_model(config, str(path))
    from peft import PeftModel
    from layer_selection import prune_attention, transformer_blocks
    model, tokenizer = load_model(config)
    prune_attention(model, range(len(transformer_blocks(model))), config.get("prune_ratio", 0.1))
    model = PeftModel.from_pretrained(model, path).merge_and_unload()
    model.eval()
    return model, tokenizer


def train_selector(config, bundle, corpora, layers, baseline_tuning, out, expected_params):
    import torch
    from layer_selection import parameter_record
    grid = config.get("tuning_grid", {"lr": [1e-5], "kl_weight": [0.1]})
    if set(grid) - {"lr", "kl_weight"}:
        raise ValueError("Tuning grid supports lr and kl_weight; epochs are a shared budget")
    trials, best = [], None
    forget = [r for r in bundle["roles"]["optimization"] if r["label"] == 1]
    for values in itertools.product(*(grid[k] for k in sorted(grid))):
        trial_config = dict(config, **dict(zip(sorted(grid), values)))
        seed_everything(config["seed"])
        model, tokenizer = load_model(config)
        model, masks, trainable = setup_student(model, layers, trial_config)
        if expected_params is not None and trainable != expected_params:
            raise ValueError(f"Unmatched trainable budget: {trainable} != {expected_params}")
        expected_params = trainable
        reference, _ = load_model(config)
        for parameter in reference.parameters():
            parameter.requires_grad = False
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                     lr=trial_config.get("lr", 1e-5), weight_decay=0.)
        logs = []
        for epoch in range(config.get("epochs", 1)):
            logs.append(unlearn_epoch(model, reference, tokenizer, forget,
                                      dict(trial_config, optimizer=optimizer), masks, epoch))
            metrics, scored = evaluate(model, tokenizer, bundle["roles"]["tuning"], corpora["tuning"],
                                       config, None, bootstrap=0)
            # Compare utility against the ORIGINAL unpruned backbone.
            in_budget = metrics["utility"]["nll"] <= baseline_tuning["utility"]["nll"] + math.log(config.get("utility_budget_mult", 1.5))
            gap = abs(metrics["attack"]["raw_auc"] - 0.5)
            trial = {"trial_index": len(trials), "lr": trial_config.get("lr", 1e-5),
                     "kl_weight": trial_config.get("kl_weight", 0.1), "epoch": epoch + 1,
                     "in_budget": in_budget, "gap": gap, "metrics": metrics,
                     "trainable_parameters": trainable, "training": deepcopy(logs)}
            trials.append(trial)
            if in_budget and (best is None or gap < best["gap"]):
                best = dict(trial, direction=frozen_direction(scored), selected_layers=layers,
                            trainable_tensors=parameter_record(model, [n for n, p in model.named_parameters() if p.requires_grad]),
                            pruned_tensors={n: int(mask.sum()) for n, mask in masks.items()})
                save_student(model, tokenizer, out / "checkpoint", trial_config)
            write_json(out / "tuning.json", {"trials": trials, "best": best})
            print(f"Tune {out.name}: epoch={epoch+1}, gap={gap:.4f}, in_budget={in_budget}", flush=True)
        del optimizer, model, reference, masks
        release()
    if best is None:
        return None, expected_params
    write_json(out / "selection.json", best)
    return best, expected_params


def check_neighbors(config, bundle):
    """Neighbors need explicit target IDs; semantics remain curator-provided."""
    all_rows = [r for rows in bundle["roles"].values() for r in rows]
    targets = {r["id"] for r in bundle["roles"]["optimization"] if r["label"] == 1}
    neighbors = canonical_records(records_from_jsonl(config["neighbors_jsonl"]), False) if config.get("neighbors_jsonl") else []
    benign = canonical_records(records_from_jsonl(config["benign_jsonl"]), False) if config.get("benign_jsonl") else []
    assert_disjoint({"experiment": all_rows, "neighbors": neighbors, "benign": benign})
    for row in neighbors:
        if row.get("target_id") not in targets or not row.get("relationship"):
            raise ValueError("Neighbors require optimization target_id and relationship description")
    return neighbors, benign


def durability(config, bundle, corpora, checkpoint, out, direction, benign, neighbors):
    # Each branch reloads the same post-unlearning checkpoint; budgets are explicit.
    from research_metrics import score_examples, summarize_scores
    branches = {}
    for kind, training_rows in (("benign", benign), ("relearning", [r for r in bundle["roles"]["optimization"] if r["label"] == 1])):
        if kind == "benign" and not training_rows:
            continue
        model, tokenizer = load_model(config, str(checkpoint))
        settings = dict(config, **config.get("durability", {}))
        log = supervised_training(model, tokenizer, training_rows, settings)
        branch = {}
        for role in ("optimization", "evaluation"):
            branch[role], _ = evaluate(model, tokenizer, bundle["roles"][role], corpora["evaluation"],
                                      config, out / kind / role, direction=direction)
        if neighbors:
            scores = score_examples(model, tokenizer, neighbors, **metric_args(config))
            branch["neighbors"] = summarize_scores(scores, bootstrap=0)
            write_json(out / kind / "neighbor_scores.json", scores)
        branch["training"] = log
        write_json(out / kind / "manifest.json", provenance(dict(settings, branch=kind), model))
        if config.get("save_durability_checkpoints", False):
            model.save_pretrained(out / kind / "checkpoint")
            tokenizer.save_pretrained(out / kind / "checkpoint")
        branches[kind] = branch
        del model
        release()
    write_json(out / "durability.json", branches)
    return branches


def run_seed(config, bundle, out):
    from layer_selection import (attention_parameters, choose_layers, parameter_record,
                                 transformer_blocks)
    from research_metrics import score_examples, summarize_scores
    seed_everything(config["seed"])
    neighbors, benign = check_neighbors(config, bundle)
    model, tokenizer = load_model(config)
    manifest = provenance(config, model)
    manifest.update(split_hash=digest(bundle), dataset_hash=bundle["dataset_hash"],
                    split_ids={role: [r["id"] for r in rows] for role, rows in bundle["roles"].items()},
                    membership_basis=bundle["membership_basis"],
                    objective="least-likely-token log-probability + token-mean KL(reference||student)")
    membership_path = Path(config["model"]) / "membership_manifest.json"
    if membership_path.exists():
        membership = read_json(membership_path)
        if membership["dataset_hash"] != bundle["dataset_hash"]:
            raise ValueError("Controlled checkpoint membership does not match the split dataset")
        manifest["membership_basis"] = membership["membership_basis"]
        manifest["membership_manifest"] = membership
    write_json(out / "manifest.json", manifest)
    corpora = utility_data(config, tokenizer, bundle)
    write_json(out / "utility_corpora.json", corpora)
    tuning, tuning_scores = evaluate(model, tokenizer, bundle["roles"]["tuning"], corpora["tuning"],
                                     config, out / "baseline" / "tuning", bootstrap=0)
    selectors = config.get("selectors", ["pruning", "masking_loss", "early", "middle", "late", "random"])
    n_layers = len(transformer_blocks(model))
    if any(s.startswith(("pruning", "masking_loss")) for s in selectors) or config.get("profile", False):
        profiles, ranks = profile(model, tokenizer, bundle, corpora, config, out / "profiles")
    else:
        profiles, ranks = [], {}
    direction = frozen_direction(tuning_scores)
    baselines = {}
    for role in ("optimization", "evaluation"):
        baselines[role], _ = evaluate(model, tokenizer, bundle["roles"][role], corpora["evaluation"],
                                     config, out / "baseline" / role, direction=direction)
    if neighbors:
        scores = score_examples(model, tokenizer, neighbors, **metric_args(config))
        baselines["neighbors"] = summarize_scores(scores, bootstrap=0)
        write_json(out / "baseline" / "neighbor_scores.json", scores)
    baselines["downstream"] = downstream(model, tokenizer, config, out / "baseline")
    write_json(out / "baseline" / "summary.json", baselines)
    del model
    release()
    results = {"baseline": baselines}
    results.update(run_baselines(config, bundle, corpora, tuning, out))
    write_json(out / "results.json", results)
    for variant in config.get("variants", ["STUDE", "LoRA-STUDE"]):
        for count in config.get("layer_counts", [4]):
            expected_params = None
            for selector in selectors:
                kind = selector.removesuffix("_window")
                records = [r for r in profiles if r["kind"] == kind]
                scores = {r["layers"][0]: r["score"] for r in records if len(r["layers"]) == 1}
                layers = choose_layers(n_layers, count, selector, scores, config["seed"], records)
                key = f"{variant}-{selector}-k{count}"
                directory = out / key
                settings = dict(config, variant=variant, selector=selector, selected_layers=layers)
                best, expected_params = train_selector(settings, bundle, corpora, layers, tuning,
                                                       directory, expected_params)
                if best is None:
                    results[key] = {"status": "no_checkpoint_within_utility_budget"}
                    write_json(out / "results.json", results)
                    continue
                model, tokenizer = load_student(directory / "checkpoint", settings)
                if variant == "LoRA-STUDE":
                    model.save_pretrained(directory / "merged_checkpoint")
                    tokenizer.save_pretrained(directory / "merged_checkpoint")
                checkpoint = directory / ("merged_checkpoint" if variant == "LoRA-STUDE" else "checkpoint")
                result = {"status": "complete", "selected_layers": layers, "trainable_parameters": best["trainable_parameters"],
                          "activation_ranks": {str(i): ranks.get(str(i)) for i in layers}}
                for role in ("optimization", "evaluation"):
                    result[role], _ = evaluate(model, tokenizer, bundle["roles"][role], corpora["evaluation"],
                                               settings, directory / role, direction=best["direction"])
                if neighbors:
                    scored = score_examples(model, tokenizer, neighbors, **metric_args(config))
                    result["neighbors"] = summarize_scores(scored, bootstrap=0)
                    write_json(directory / "neighbor_scores.json", scored)
                result["downstream"] = downstream(model, tokenizer, config, directory)
                record = provenance(settings, model)
                record.update(split_hash=digest(bundle), best_trial=best,
                              deployment_attention=parameter_record(model, attention_parameters(model)),
                              pruning_at_deployment="merged_adapter_may_refill_zeros" if variant == "LoRA-STUDE" else "zeros_preserved",
                              matched_group=f"{variant}-k{count}",
                              tuning_candidates=len(read_json(directory / "tuning.json")["trials"]),
                              checkpoint=str(checkpoint.resolve()))
                write_json(directory / "manifest.json", record)
                del model
                release()
                if config.get("durability"):
                    result["durability"] = durability(settings, bundle, corpora, checkpoint,
                                                       directory / "durability", best["direction"], benign, neighbors)
                results[key] = result
                write_json(out / "results.json", results)
    write_json(out / "rank_and_utility_analysis.json",
               rank_and_utility_analysis(profiles, results, baselines, config.get("utility_match_tolerance", 0.05)))
    return results


def aggregate_runs(paths, out):
    import numpy as np
    groups = {}
    for path in paths:
        for method, result in read_json(path).items():
            if result.get("status") != "complete":
                continue
            metrics = result["evaluation"]
            values = {"oriented_auc": metrics["attack"]["oriented_auc"],
                      "max_auc": metrics["attack"]["max_auc"], "utility_nll": metrics["utility"]["nll"]}
            if metrics.get("extraction"):
                values["exact_match"] = metrics["extraction"]["exact_match"]
            groups.setdefault(method, []).append(values)
    result = {}
    for method, rows in groups.items():
        result[method] = {key: {"n_seeds": len(values), "mean": float(np.mean(values)),
                                 "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                                 "values": values}
                          for key in rows[0] for values in [[r[key] for r in rows if key in r]]}
    write_json(out, result)


def rank_and_utility_analysis(profiles, results, baseline, tolerance=0.05):
    """Descriptive associations and nearby utility pairs; never select checkpoints here."""
    import numpy as np
    def average_ranks(values):
        values = np.asarray(values)
        order = np.argsort(values, kind="stable")
        ranked = np.empty(len(values), dtype=float)
        start = 0
        while start < len(values):
            stop = start + 1
            while stop < len(values) and values[order[stop]] == values[order[start]]:
                stop += 1
            ranked[order[start:stop]] = (start + stop - 1) / 2
            start = stop
        return ranked
    def association(x, y):
        if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
            return {"n": len(x), "pearson": None, "spearman": None}
        return {"n": len(x), "pearson": float(np.corrcoef(x, y)[0, 1]),
                "spearman": float(np.corrcoef(average_ranks(x), average_ranks(y))[0, 1])}
    correlations = {}
    for kind in ("pruning", "masking_loss"):
        rows = [r for r in profiles if r["kind"] == kind and len(r["layers"]) == 1]
        x = [next(iter(r["activation_ranks"].values()))["effective_rank"] for r in rows]
        correlations[kind] = {key: association(x, [r[key] for r in rows])
                              for key in ("member_nll_delta", "auc_reduction", "utility_nll_delta")}
    rank_results = [(name, r) for name, r in results.items() if r.get("status") == "complete"
                    and r.get("activation_ranks") and all(r["activation_ranks"].values())]
    for variant in ("STUDE", "LoRA-STUDE"):
        rows = [r for name, r in rank_results if name.startswith(variant + "-")]
        x = [np.mean([v["effective_rank"] for v in r["activation_ranks"].values()]) for r in rows]
        correlations[variant + "_unlearning"] = association(x, [baseline["evaluation"]["attack"]["max_auc"] - r["evaluation"]["attack"]["max_auc"] for r in rows])
    complete = [(name, r) for name, r in results.items() if r.get("status") == "complete"]
    pairs = []
    for (a, ra), (b, rb) in itertools.combinations(complete, 2):
        delta = abs(ra["evaluation"]["utility"]["nll"] - rb["evaluation"]["utility"]["nll"])
        if delta <= tolerance:
            pairs.append({"methods": [a, b], "utility_nll_distance": delta,
                          "max_auc": [ra["evaluation"]["attack"]["max_auc"], rb["evaluation"]["attack"]["max_auc"]]})
    return {"correlations": correlations, "utility_matched_pairs": pairs, "utility_nll_tolerance": tolerance,
            "interpretation": "Descriptive associations; overlapping regions/seeds are not independent causal evidence. No final-evaluation retuning."}


def build_neighbors(args):
    import re
    bundle = validate_splits(read_json(args.splits))
    targets = [r for r in bundle["roles"]["optimization"] if r["label"] == 1]
    candidates = canonical_records(records_from_jsonl(args.candidates), False)
    existing = {r["content_id"] for rows in bundle["roles"].values() for r in rows}
    candidates = [r for r in candidates if r["content_id"] not in existing]
    def words(text):
        return set(re.findall(r"\w+", text.lower()))
    pool = [(r, words(r["input"])) for r in candidates]
    selected, used = [], set()
    for target in targets:
        query = words(target["input"])
        ranked = sorted(((len(query & tokens) / max(1, len(query | tokens)), row) for row, tokens in pool),
                        key=lambda item: (-item[0], item[1]["id"]))
        accepted = 0
        for score, row in ranked:
            if score < args.min_similarity or row["id"] in used:
                continue
            selected.append(dict(row, target_id=target["id"], relationship="lexical_Jaccard_neighbor",
                                 similarity=score))
            used.add(row["id"])
            accepted += 1
            if accepted >= args.per_target:
                break
    if not selected:
        raise ValueError("No neighbors met the requested similarity threshold")
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    import json
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected))
    print(f"Saved {len(selected)} lexical neighbors; semantic relevance needs curator review")


def audit(args):
    audits = []
    for filename in args.paths:
        data = read_json(filename)
        manifest = data if "config" in data and "schema_version" in data else None
        issues = []
        if manifest is None:
            issues.append("No experiment manifest: backbone, variant, budgets, splits, and seed cannot be certified")
        else:
            for key in ("git_commit", "config_hash"):
                if key not in manifest:
                    issues.append(f"Missing {key}")
            if "split_hash" not in manifest and "dataset_hash" not in manifest:
                issues.append("Missing dataset/split fingerprint")
            if manifest.get("config_hash") != digest(manifest["config"]):
                issues.append("Configuration hash mismatch")
        audits.append({"file": filename, "issues": issues,
                       "rerun_required": bool(issues), "stored_entries": len(data)})
    write_json(args.output, audits)
    print(f"Audit saved to {args.output}")


def run(args):
    config = read_json(args.config)
    if config.get("cpu_threads"):
        import torch
        torch.set_num_threads(config["cpu_threads"])
    if not config.get("seeds", [42]) or len(set(config.get("seeds", [42]))) != len(config.get("seeds", [42])):
        raise ValueError("Provide distinct, nonempty seeds")
    if config.get("epochs", 1) < 1 or config.get("utility_budget_mult", 1.5) <= 0:
        raise ValueError("Invalid epoch count or utility budget")
    if not 0 <= config.get("objective_min_k", 0.) <= 1:
        raise ValueError("objective_min_k must be between zero and one")
    bundle = validate_splits(read_json(config["splits"]))
    config["split_seed"] = bundle["seed"]
    outputs = []
    for seed in config.get("seeds", [42]):
        out = Path(config["output"]) / f"seed-{seed}"
        if out.exists():
            raise FileExistsError(f"Refusing to overwrite experiment: {out}")
        out.mkdir(parents=True)
        settings = dict(config, seed=seed)
        try:
            run_seed(settings, bundle, out)
        except Exception as exc:
            write_json(out / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
            raise
        outputs.append(out / "results.json")
    aggregate_runs(outputs, Path(config["output"]) / "aggregate.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="Create immutable, checked experiment splits")
    p.add_argument("--input", help="Local JSONL with input/text, label, optional id")
    p.add_argument("--length", type=int, default=128)
    p.add_argument("--revision", default="main")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fractions", nargs=4, type=float, default=[0.2, 0.2, 0.4, 0.2])
    p.add_argument("--output", required=True)
    p.set_defaults(function=prepare)
    for name, function in (("controlled", controlled), ("run", run)):
        p = commands.add_parser(name)
        p.add_argument("--config", required=True)
        p.set_defaults(function=function)
    p = commands.add_parser("neighbors", help="Build disjoint lexical neighbors for curator review")
    p.add_argument("--splits", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--per-target", type=int, default=1)
    p.add_argument("--min-similarity", type=float, default=0.1)
    p.set_defaults(function=build_neighbors)
    p = commands.add_parser("audit", help="Check existing JSON result provenance")
    p.add_argument("paths", nargs="+")
    p.add_argument("--output", required=True)
    p.set_defaults(function=audit)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
