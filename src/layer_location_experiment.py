"""Rebuttal control: does STUDE's selected layer band matter?

Compares the dev-selected STUDE band against early, late, random contiguous,
and random scattered layer sets of the same width, under one matched protocol:

  * same width  -> identical trainable-parameter count and module types
                   (full transformer blocks, as in STUDE; asserted per run)
  * same objective as train_model.py / paper Table 3: 10% L1 pruning of all
    attention Q/K/V, then least-likely-token ascent + token-summed
    KL(reference || student) on the forget sequences, Adafactor, grad-accum 32
  * same data, same number of optimizer steps, same tuning grid per location
  * several training seeds, several random draws per random arm

Data roles (no decision ever reads the test half):
  split        WikiMIA (length L) is split 50/50, stratified by label, into
               DEV and TEST with a fixed split seed.
  forget set   all members (dev + test) are unlearned, as in the paper.
  band choice  `select-band` scans every contiguous window of the given width,
               prunes it, and picks the largest Min-K ROC-AUC drop on DEV only.
  model choice per run, the saved/selected epoch and the tuning-grid point
               are chosen by DEV Min-K AUC closest to 0.5, subject to DEV
               utility PPL (WikiText-2 *validation*) <= budget x original.
  reporting    TEST pairs (test members vs test non-members) and WikiText-2
               *test* PPL. Test metrics are logged every epoch but never used
               for selection.

Stages (run in order):
  split -> select-band -> plan -> train (one run per plan index; job-array
  friendly, resumable) -> summarize
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from experiment_protocol import provenance, read_json, seed_everything, text_id, write_json  # noqa: E402


# ── Shared helpers ────────────────────────────────────────────────────────────

def load_utility_lines(split, min_chars=64):
    from datasets import load_dataset
    # Canonical Hub id; newer `datasets` releases reject the bare "wikitext" alias.
    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    return [t.strip() for t in wt["text"] if len(t.strip()) >= min_chars]


def load_lm(name, dtype="bfloat16", device=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=getattr(torch, dtype)).to(device)
    model.eval()
    return model, tok


def mink_auc(model, tok, rows, k=0.2):
    """Min-K% ROC-AUC only (the selection signal); cheaper than the full panel."""
    import torch
    from sklearn.metrics import roc_auc_score
    from memorization_effect import get_token_logprobs, min_k_percent_score
    was_training = model.training
    model.eval()
    with torch.no_grad():
        scores = [min_k_percent_score(get_token_logprobs(r["input"], tok, model), k=k) for r in rows]
    model.train(was_training)
    return float(roc_auc_score([r["label"] for r in rows], scores))


def full_panel(model, tok, rows):
    """Min-K ROC-AUC, MK-Gap, PPL/Lowercase/Zlib AUCs (no reference model)."""
    import torch
    from memorization_effect import estimate_memorization
    was_training = model.training
    model.eval()
    with torch.no_grad():
        result = estimate_memorization(model, rows, tok)
    model.train(was_training)
    return {k: v for k, v in result.items() if v is not None}


def utility(model, tok, lines, max_length):
    from train_model import evaluate_utility
    return evaluate_utility(model, tok, lines, max_length)


def layer_of(name):
    from layer_selection import layer_index
    return layer_index(name)


# ── Stage 1: split ────────────────────────────────────────────────────────────

def cmd_split(args):
    from datasets import load_dataset
    ds = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{args.length}")
    rows, seen = [], set()
    for ex in ds:
        rid = text_id(ex["input"])
        if rid in seen:
            continue
        seen.add(rid)
        rows.append({"id": rid, "input": ex["input"], "label": int(ex["label"])})
    rng = random.Random(args.split_seed)
    dev, test = [], []
    for label in (0, 1):
        group = sorted([r for r in rows if r["label"] == label], key=lambda r: r["id"])
        rng.shuffle(group)
        cut = round(len(group) * args.dev_fraction)
        dev += group[:cut]
        test += group[cut:]
    bundle = {"source": f"swj0419/WikiMIA:WikiMIA_length{args.length}",
              "dataset_fingerprint": getattr(ds, "_fingerprint", None),
              "split_seed": args.split_seed, "dev_fraction": args.dev_fraction,
              "dev": sorted(dev, key=lambda r: r["id"]), "test": sorted(test, key=lambda r: r["id"])}
    out = Path(args.out)
    if out.exists() and read_json(out) != bundle:
        raise FileExistsError(f"{out} exists with different content; choose a new path")
    write_json(out, bundle)
    for role in ("dev", "test"):
        n1 = sum(r["label"] for r in bundle[role])
        print(f"{role}: {len(bundle[role])} rows ({n1} members, {len(bundle[role]) - n1} non-members)")


# ── Stage 2: select-band (DEV only) ───────────────────────────────────────────

def cmd_select_band(args):
    import torch
    from layer_selection import attention_parameters, transformer_blocks
    from pruner import prune_l1_unstructured
    splits = read_json(args.splits)
    seed_everything(0)
    model, tok = load_lm(args.model, dtype="float32")
    n_layers = len(transformer_blocks(model))
    width = args.width
    if not 1 <= width <= n_layers:
        raise ValueError("width outside model depth")

    # Original (unpruned) reference numbers shared by every run: utility budget
    # and the denominators for the gap-closed ratio.
    print("Original model: dev/test panels and utilities")
    original = {"dev_mink_auc": mink_auc(model, tok, splits["dev"]),
                "test_panel": full_panel(model, tok, splits["test"]),
                "utility_dev_ppl": utility(model, tok, load_utility_lines("validation"), args.max_length),
                "utility_test_ppl": utility(model, tok, load_utility_lines("test"), args.max_length)}
    print(json.dumps(original, indent=2))

    windows = []
    for start in range(n_layers - width + 1):
        layers = list(range(start, start + width))
        params = attention_parameters(model, layers)
        backup = {n: p.detach().clone() for n, p in params.items()}
        prune_l1_unstructured(model, prune_ratio=args.prune_ratio, layer_start=start, layer_end=start + width - 1)
        auc = mink_auc(model, tok, splits["dev"])
        with torch.no_grad():
            for n, p in params.items():
                p.copy_(backup[n])
        del backup
        drop = original["dev_mink_auc"] - auc
        windows.append({"start": start, "layers": layers, "dev_mink_auc": auc, "auc_drop": drop})
        print(f"window {start:2d}-{start + width - 1:2d}: dev AUC {auc:.4f} (drop {drop:+.4f})", flush=True)
    best = max(windows, key=lambda w: (w["auc_drop"], -w["start"]))
    record = {"model": args.model, "width": width, "n_layers": n_layers, "prune_ratio": args.prune_ratio,
              "criterion": "largest dev Min-K ROC-AUC drop when only this window is L1-pruned",
              "selected_layers": best["layers"], "windows": windows, "original": original,
              "max_length": args.max_length,
              "provenance": provenance({k: v for k, v in vars(args).items() if k != "fn"}, model)}
    write_json(args.out, record)
    print(f"Selected band (dev): layers {best['layers'][0]}-{best['layers'][-1]}")


# ── Stage 3: plan ─────────────────────────────────────────────────────────────

def cmd_plan(args):
    band = read_json(args.band)
    n, w = band["n_layers"], band["width"]
    named = {"selected": band["selected_layers"]}
    if args.paper_band:
        lo, hi = args.paper_band
        if hi - lo + 1 != w:
            raise ValueError("paper band width must equal --width used in select-band")
        named["paper_band"] = list(range(lo, hi + 1))
    named.update(early=list(range(w)), late=list(range(n - w, n)))
    # Identical layer sets are trained once; the later name is reported as an alias.
    arms, aliases = [], {}
    for name, layers in named.items():
        twin = next((a["arm"] for a in arms if a["layers"] == layers), None)
        if twin:
            aliases[name] = twin
        else:
            arms.append({"arm": name, "draw": 0, "layers": layers})

    used_starts = {v[0] for v in named.values() if v == list(range(v[0], v[0] + w))}
    starts = [s for s in range(n - w + 1) if s not in used_starts]
    rng = random.Random(args.draw_seed)
    for d, s in enumerate(rng.sample(starts, min(args.random_draws, len(starts)))):
        arms.append({"arm": "random_contiguous", "draw": d, "layers": list(range(s, s + w))})
    d = 0
    while d < args.random_draws:
        layers = sorted(rng.sample(range(n), w))
        if layers == list(range(layers[0], layers[0] + w)):
            continue  # contiguous by chance; scattered arm must not be a band
        arms.append({"arm": "random_scattered", "draw": d, "layers": layers})
        d += 1

    runs = []
    for arm in arms:
        for seed in args.seeds:
            for lr in args.lrs:
                for kl in args.kl_weights:
                    rid = f"{arm['arm']}-d{arm['draw']}__s{seed}__lr{lr:g}_kl{kl:g}"
                    runs.append(dict(arm, run_id=rid, seed=seed, lr=lr, kl_weight=kl))
    plan = {"band": str(Path(args.band).resolve()), "splits": str(Path(args.splits).resolve()),
            "model": band["model"], "width": w, "prune_ratio": band["prune_ratio"],
            "epochs": args.epochs, "grad_accum": args.grad_accum, "max_length": band["max_length"],
            "budget_mult": args.budget_mult, "tuning_grid": {"lr": args.lrs, "kl_weight": args.kl_weights},
            "seeds": args.seeds, "arms": arms, "aliases": aliases, "runs": runs}
    write_json(args.out, plan)
    print(f"{len(arms)} layer sets x {len(args.seeds)} seeds x "
          f"{len(args.lrs) * len(args.kl_weights)} grid points = {len(runs)} runs")
    for a in arms:
        print(f"  {a['arm']:<18} draw {a['draw']}: {a['layers']}")
    for name, twin in aliases.items():
        print(f"  {name:<18} = same layers as {twin} (not trained twice)")


# ── Stage 4: train one run ────────────────────────────────────────────────────

def dev_choice(epochs, budget):
    """Best epoch by DEV only: in utility budget, Min-K AUC closest to 0.5."""
    ok = [e for e in epochs if e["epoch"] >= 1 and math.isfinite(e["utility_dev_ppl"])
          and e["utility_dev_ppl"] <= budget]
    return min(ok, key=lambda e: (abs(e["dev_mink_auc"] - 0.5), e["epoch"])) if ok else None


def cmd_train(args):
    import torch
    import torch.nn.functional as F
    from transformers import get_linear_schedule_with_warmup
    from transformers.optimization import Adafactor
    from pruner import prune_l1_unstructured
    from train_model import min_k_percent_loss

    plan = read_json(args.plan)
    run = plan["runs"][args.index]
    out = Path(args.out) / "runs" / run["run_id"]
    if (out / "result.json").exists():
        print(f"{run['run_id']} already complete; skipping")
        return
    band, splits = read_json(plan["band"]), read_json(plan["splits"])
    original = band["original"]
    budget = plan["budget_mult"] * original["utility_dev_ppl"]
    layers = set(run["layers"])
    seed_everything(run["seed"])
    print(f"Run {args.index}: {run['run_id']} layers={run['layers']}", flush=True)

    model, tok = load_lm(plan["model"], dtype="bfloat16")   # matches train_model.py
    prune_l1_unstructured(model, prune_ratio=plan["prune_ratio"], layer_start=0)
    pruning_masks = {n: (p.data == 0).clone() for n, p in model.named_parameters() if (p.data == 0).any()}

    per_layer = {}
    for name, p in model.named_parameters():
        idx = layer_of(name)
        p.requires_grad = idx is not None and idx in layers
        if idx is not None:
            per_layer[idx] = per_layer.get(idx, 0) + p.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected = sum(per_layer[i] for i in layers)
    if trainable != expected or len(set(per_layer.values())) != 1:
        raise RuntimeError(f"Unmatched trainable budget: {trainable} vs {expected}")
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.train()

    ref, _ = load_lm(plan["model"], dtype="bfloat16")
    for p in ref.parameters():
        p.requires_grad = False

    forget = [r for r in splits["dev"] + splits["test"] if r["label"] == 1]
    accum, epochs = plan["grad_accum"], plan["epochs"]
    steps_per_epoch = math.ceil(len(forget) / accum)
    total_steps = steps_per_epoch * epochs   # includes the flush step every epoch
    optimizer = Adafactor([p for p in model.parameters() if p.requires_grad], lr=run["lr"],
                          relative_step=False, scale_parameter=False, warmup_init=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    util_dev, util_test = load_utility_lines("validation"), load_utility_lines("test")

    def step():
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        for name, p in model.named_parameters():
            if name in pruning_masks and p.grad is not None:
                p.grad.data[pruning_masks[name]] = 0.0
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    def measure(epoch, train_log):
        rec = {"epoch": epoch, "dev_mink_auc": mink_auc(model, tok, splits["dev"]),
               "utility_dev_ppl": utility(model, tok, util_dev, plan["max_length"]),
               "test_panel": full_panel(model, tok, splits["test"]),
               "utility_test_ppl": utility(model, tok, util_test, plan["max_length"])}
        rec.update(train_log)
        print(f"epoch {epoch}: dev AUC {rec['dev_mink_auc']:.4f} | dev PPL {rec['utility_dev_ppl']:.2f} "
              f"(budget {budget:.2f}) | test AUC {rec['test_panel']['roc_auc']:.4f}", flush=True)
        return rec

    history = [measure(0, {"note": "pruned, before unlearning"})]
    optimizer_steps, best_gap = 0, None
    for epoch in range(epochs):
        order = list(forget)
        random.Random(run["seed"] * 1000 + epoch).shuffle(order)
        optimizer.zero_grad()
        ascent_sum, kl_sum, n_used = 0.0, 0.0, 0
        for i, row in enumerate(order):
            enc = tok(row["input"], return_tensors="pt", truncation=True, max_length=plan["max_length"])
            ids = enc["input_ids"].to(model.device)
            mask = enc["attention_mask"].to(model.device)
            if ids.size(1) >= 2:
                logits = model(input_ids=ids, attention_mask=mask).logits
                ascent = min_k_percent_loss(logits, ids, k=0)       # NLL of least-likely token
                if torch.isfinite(ascent):
                    with torch.no_grad():
                        ref_lp = F.log_softmax(ref(input_ids=ids, attention_mask=mask).logits[:, :-1].float(), -1)
                    cur_lp = F.log_softmax(logits[:, :-1].float(), -1)
                    kl = F.kl_div(cur_lp, ref_lp.exp(), reduction="batchmean")  # token-sum, as train_model.py
                    ((-ascent + run["kl_weight"] * kl) / accum).backward()
                    ascent_sum += ascent.item()
                    kl_sum += kl.item()
                    n_used += 1
            if (i + 1) % accum == 0 or i + 1 == len(order):
                step()
                optimizer_steps += 1
        rec = measure(epoch + 1, {"optimizer_steps_total": optimizer_steps, "examples_used": n_used,
                                  "mean_min_token_nll": ascent_sum / max(n_used, 1),
                                  "mean_kl": kl_sum / max(n_used, 1)})
        history.append(rec)
        if args.save_dir:
            choice = dev_choice(history, budget)
            if choice is rec and (best_gap is None or abs(rec["dev_mink_auc"] - 0.5) < best_gap):
                best_gap = abs(rec["dev_mink_auc"] - 0.5)
                model.save_pretrained(Path(args.save_dir) / run["run_id"])
                tok.save_pretrained(Path(args.save_dir) / run["run_id"])

    result = {"run": run, "trainable_parameters": trainable, "forget_examples": len(forget),
              "optimizer_steps": optimizer_steps, "utility_budget_dev_ppl": budget,
              "history": history, "dev_choice_epoch": (dev_choice(history, budget) or {}).get("epoch"),
              "provenance": provenance(dict(plan_path=str(Path(args.plan).resolve()), index=args.index), model)}
    write_json(out / "result.json", result)
    print(f"Saved {out / 'result.json'}")


# ── Stage 5: summarize ────────────────────────────────────────────────────────

def cmd_summarize(args):
    import numpy as np
    plan = read_json(args.plan)
    band = read_json(plan["band"])
    original = band["original"]
    base_auc = original["test_panel"]["roc_auc"]
    budget = plan["budget_mult"] * original["utility_dev_ppl"]
    runs_dir = Path(args.out) / "runs"

    # Per (arm, draw, seed): pick grid point + epoch by DEV only, report TEST.
    groups, missing = {}, []
    for run in plan["runs"]:
        path = runs_dir / run["run_id"] / "result.json"
        if not path.exists():
            missing.append(run["run_id"])
            continue
        res = read_json(path)
        choice = dev_choice(res["history"], budget)
        key = (run["arm"], run["draw"], run["seed"])
        groups.setdefault(key, [])
        if choice is not None:
            groups[key].append((abs(choice["dev_mink_auc"] - 0.5), run, choice, res["trainable_parameters"]))
    selections, params = [], set()
    for (arm, draw, seed), cands in sorted(groups.items()):
        if not cands:
            selections.append({"arm": arm, "draw": draw, "seed": seed, "status": "no_in_budget_config"})
            continue
        _, run, ch, n_params = min(cands, key=lambda c: (c[0], c[1]["lr"], c[1]["kl_weight"]))
        params.add(n_params)
        t = ch["test_panel"]
        gap = abs(t["roc_auc"] - 0.5)
        selections.append({"arm": arm, "draw": draw, "seed": seed, "status": "ok", "layers": run["layers"],
                           "lr": run["lr"], "kl_weight": run["kl_weight"], "epoch": ch["epoch"],
                           "dev_mink_auc": ch["dev_mink_auc"], "test_auc": t["roc_auc"], "test_gap": gap,
                           "rho": 1 - gap / abs(base_auc - 0.5), "test_mk_gap": t["mink_gap"],
                           "test_ppl_auc": t["ppl_auc"], "test_lowercase_auc": t["lowercase_auc"],
                           "test_zlib_auc": t["zlib_auc"], "utility_test_ppl": ch["utility_test_ppl"],
                           "trainable_parameters": n_params})
    metrics = ["test_auc", "test_gap", "rho", "test_mk_gap", "test_ppl_auc", "test_lowercase_auc",
               "test_zlib_auc", "utility_test_ppl"]
    order = ["selected", "paper_band", "early", "late", "random_contiguous", "random_scattered"]
    table = {}
    for arm in [a for a in order if any(s["arm"] == a for s in selections)]:
        ok = [s for s in selections if s["arm"] == arm and s["status"] == "ok"]
        table[arm] = {"n_ok": len(ok), "n_failed": sum(1 for s in selections if s["arm"] == arm) - len(ok)}
        for m in metrics:
            vals = [s[m] for s in ok]
            table[arm][m] = {"mean": float(np.mean(vals)) if vals else None,
                             "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None}
    for name, twin in plan.get("aliases", {}).items():
        if twin in table:
            table[name] = dict(table[twin], alias_of=twin)
    table = {a: table[a] for a in order if a in table}
    summary = {"original_test_auc": base_auc, "original_utility_test_ppl": original["utility_test_ppl"],
               "dev_utility_budget_ppl": budget, "selected_band": band["selected_layers"],
               "trainable_parameters_per_run": sorted(params), "matched_parameter_budget": len(params) <= 1,
               "missing_runs": missing, "per_arm": table, "selections": selections}
    write_json(Path(args.out) / "summary.json", summary)

    def fmt(cell):
        if cell["mean"] is None:
            return "n/a"
        return f"{cell['mean']:.3f}" + (f" ± {cell['std']:.3f}" if cell["std"] is not None else "")
    lines = [f"Selected band (dev): {band['selected_layers']}  |  original test AUC {base_auc:.3f}, "
             f"test PPL {original['utility_test_ppl']:.2f}  |  trainable params/run: {sorted(params)}", "",
             "| Arm | n | Test AUC | ρ (gap closed) | MK-Gap | PPL-AUC | Lowercase | Zlib | Test PPL |",
             "|---|---|---|---|---|---|---|---|---|"]
    for arm, row in table.items():
        label = arm + (f" (= {row['alias_of']})" if row.get("alias_of") else "")
        lines.append(f"| {label} | {row['n_ok']}" + (f" (+{row['n_failed']} over budget)" if row["n_failed"] else "")
                     + " | " + " | ".join(fmt(row[m]) for m in
                                          ["test_auc", "rho", "test_mk_gap", "test_ppl_auc",
                                           "test_lowercase_auc", "test_zlib_auc", "utility_test_ppl"]) + " |")
    if missing:
        lines += ["", f"WARNING: {len(missing)} planned runs have no result yet."]
    (Path(args.out) / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("split")
    p.add_argument("--length", type=int, default=64)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--dev-fraction", type=float, default=0.5)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_split)

    p = sub.add_parser("select-band")
    p.add_argument("--model", default="EleutherAI/pythia-2.8b")
    p.add_argument("--splits", required=True)
    p.add_argument("--width", type=int, default=15, help="Band width (paper: layers 10-24 = 15)")
    p.add_argument("--prune-ratio", type=float, default=0.10)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_select_band)

    p = sub.add_parser("plan")
    p.add_argument("--band", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--paper-band", type=int, nargs=2, metavar=("FIRST", "LAST"), default=[10, 24],
                   help="Also run the paper's fixed band for reference (default 10 24)")
    p.add_argument("--random-draws", type=int, default=3)
    p.add_argument("--draw-seed", type=int, default=1234)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--lrs", type=float, nargs="+", default=[1e-5, 2e-5, 4e-5])
    p.add_argument("--kl-weights", type=float, nargs="+", default=[0.7])
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=32)
    p.add_argument("--budget-mult", type=float, default=1.5)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("train")
    p.add_argument("--plan", required=True)
    p.add_argument("--index", type=int, required=True, help="Plan run index (e.g. $SLURM_ARRAY_TASK_ID)")
    p.add_argument("--out", required=True)
    p.add_argument("--save-dir", help="Optionally save the dev-selected checkpoint (for LM-Eval)")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("summarize")
    p.add_argument("--plan", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_summarize)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
