import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from transformers.optimization import Adafactor
from datasets import load_dataset
from tqdm import tqdm
from pruner import prune_l1_unstructured
from memorization_effect import calculate_perplexity, estimate_memorization
import json
import numpy as np


def min_k_percent_loss(logits, input_ids, k=0.2):
    """Min-K% loss: negative mean of the bottom k% token log-probabilities.

    Minimizing this loss pushes the model to assign higher probability
    to its least-confident tokens — the exact tokens Min-K% measures.
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1).squeeze(0)  # [seq_len - 1]

    n = token_log_probs.numel()
    k_n = max(1, int(n * k))
    bottom_k = torch.topk(token_log_probs, k_n, largest=False).values

    # loss = -mean(bottom_k_log_probs)
    # log_probs are negative, so negating gives a positive loss
    # minimizing this maximizes the Min-K% score
    return -bottom_k.mean()


def measure_accuracy(model, tokenizer, device, num_samples=200):
    """Compute next-token accuracy on LAMBADA."""
    lambada = load_dataset("EleutherAI/lambada_openai", split="test")
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i, example in enumerate(lambada):
            if i >= num_samples:
                break
            full_ids = tokenizer(example["text"], return_tensors="pt")["input_ids"][0]
            if len(full_ids) < 2:
                continue
            context = full_ids[:-1].unsqueeze(0).to(device)
            target = full_ids[-1].item()
            logits = model(input_ids=context).logits[0, -1, :]
            if logits.argmax().item() == target:
                correct += 1
            total += 1
    model.train()
    return correct / total if total > 0 else 0.0


def load_utility_texts(n=64, min_chars=64):
    """Held-out wikitext-2 test lines — a corpus training never touches."""
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t.strip() for t in wt["text"] if len(t.strip()) >= min_chars]
    return texts[:n]


def evaluate_utility(model, tokenizer, texts, max_length=128):
    """Mean held-out perplexity — the utility signal that should stay near baseline.

    Toggles the model to eval() for the measurement and restores train() after,
    so it can be called between epochs without disturbing the training state.
    """
    was_training = model.training
    model.eval()
    ppls = []
    with torch.no_grad():
        for t in texts:
            p = calculate_perplexity(t, model, tokenizer, max_length=max_length)
            if np.isfinite(p):
                ppls.append(p)
    if was_training:
        model.train()
    return float(np.mean(ppls)) if ppls else float("nan")


def evaluate_auc(model, tokenizer, eval_dataset):
    """WikiMIA Min-K% membership-inference ROC-AUC — the forgetting signal.

    Toggles to eval() for the measurement and restores train() after. No ref
    model needed: the ROC-AUC is computed from the Min-K% score alone.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        res = estimate_memorization(model, eval_dataset, tokenizer)
    if was_training:
        model.train()
    return res["roc_auc"]


def get_layer_prefix(model):
    """Return the parameter name prefix used for transformer layers in this model."""
    for name, _ in model.named_parameters():
        if "gpt_neox.layers." in name:
            return "gpt_neox.layers."
        if "model.layers." in name:
            return "model.layers."
        if "transformer.h." in name:      # GPT-Neo / GPT-2 style
            return "transformer.h."
    raise ValueError(f"Cannot detect layer prefix for {type(model).__name__}")


def lora_targets(model):
    """(target_modules, layers_pattern) for attaching LoRA to attention projections.

    peft matches `target_modules` by suffix and `layers_pattern` (the layer-list
    attribute name) with `layers_to_transform` to restrict adapters to a layer band.
    """
    names = [n for n, _ in model.named_parameters()]
    if any("query_key_value" in n for n in names):   # GPT-NeoX / Pythia (fused QKV)
        return ["query_key_value"], "layers"
    if any("transformer.h." in n for n in names):    # GPT-Neo
        return ["q_proj", "k_proj", "v_proj"], "h"
    if any("model.layers." in n for n in names):      # Llama / Mistral
        return ["q_proj", "k_proj", "v_proj"], "layers"
    raise ValueError(f"Cannot determine LoRA targets for {type(model).__name__}")


def measure_gradient_norms(model, tokenizer, dataset, device, num_samples=16):
    """Run gradient ascent backward on num_samples and return mean grad norm per layer."""
    prefix = get_layer_prefix(model)
    model.zero_grad()
    layer_grads = {}

    for i, example in enumerate(dataset):
        if i >= num_samples:
            break
        inputs = tokenizer(example["input"], return_tensors="pt",
                           truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        if input_ids.size(1) < 2:
            continue

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = min_k_percent_loss(outputs.logits, input_ids, k=0)
        if torch.isfinite(loss):
            (-loss).backward()

    for name, param in model.named_parameters():
        if param.grad is not None and prefix in name:
            layer_idx = int(name.split(prefix)[1].split(".")[0])
            layer_grads.setdefault(layer_idx, []).append(param.grad.norm().item())

    model.zero_grad()
    return {layer: float(np.mean(norms)) for layer, norms in sorted(layer_grads.items())}


def train_model(
    model_name="EleutherAI/pythia-2.8b",
    dataset_length=64,
    epochs=3,
    lr=5e-5,
    max_length=128,
    gradient_accumulation_steps=16,
    save_path="trained_model",
    max_steps=None,
    kl_weight=0.0,
    budget_mult=1.5,
    use_peft=False,
    lora_rank=8,
    lora_alpha=16,
    lora_dropout=0.0,
):
    """Fine-tune Pythia on WikiMIA true positives using gradient ascent on the single minimum token.

    Only trains on label=1 (member/memorized) samples. We do gradient ascent
    on the single least-confident token (k_n=1) to sharpen the memorization trough —
    pushing that token's log-prob down to widen the gap between member and non-member Min-K% scores.
    A KL penalty against the frozen reference model prevents general capability from drifting.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    )
    model = model.to(device)

    num_layers = model.config.num_hidden_layers
    # Train only layers 10–24; freeze embeddings, layers 0–9, and layers 25–31
    TRAIN_START, TRAIN_END = 10, 24

    # Measure before pruning
    print("Measuring gradient norms and accuracy before pruning...")
    member_dataset = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{dataset_length}")
    member_dataset = member_dataset.filter(lambda ex: ex["label"] == 1)
    grads_before = measure_gradient_norms(model, tokenizer, member_dataset, device)
    acc_before = measure_accuracy(model, tokenizer, device)
    print(f"Accuracy before pruning: {acc_before*100:.2f}%")

    # Step 1: 10% L1 unstructured magnitude pruning across ALL layers
    print(f"Applying 10% L1 unstructured pruning to ALL layers (0–{num_layers-1})...")
    prune_l1_unstructured(model, prune_ratio=0.10, layer_start=0)

    # Measure after pruning
    print("Measuring gradient norms and accuracy after pruning...")
    grads_after = measure_gradient_norms(model, tokenizer, member_dataset, device)
    acc_after = measure_accuracy(model, tokenizer, device)
    print(f"Accuracy after pruning: {acc_after*100:.2f}%")

    # Dump to JSON
    with open("gradient_flow.json", "w") as f:
        json.dump({
            "before_pruning": {"gradient_norms": grads_before, "accuracy": acc_before},
            "after_pruning":  {"gradient_norms": grads_after,  "accuracy": acc_after},
        }, f, indent=2)
    print("Gradient flow + accuracy data saved to gradient_flow.json")

    # Snapshot the 10%-pruned positions so we can keep them zero during full-FT
    # and re-impose sparsity after merging LoRA adapters. prune.remove() bakes the
    # mask into the weight, leaving zeros that would otherwise get re-grown.
    layer_prefix = get_layer_prefix(model)
    pruning_masks = {}
    for name, param in model.named_parameters():
        zeros = (param.data == 0)
        if zeros.any():
            pruning_masks[name] = zeros.clone()

    # Step 2: localize training to the center layers. Full-FT trains all params in
    # layers 10–24; PEFT instead attaches LoRA adapters to the attention projections
    # in that band and trains only those (far fewer parameters).
    if use_peft:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise ImportError("The --peft path needs the peft package: `pip install peft`")
        target_modules, layers_pattern = lora_targets(model)
        lora_cfg = LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=target_modules,
            layers_to_transform=list(range(TRAIN_START, TRAIN_END + 1)),
            layers_pattern=layers_pattern,
            bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        print(f"PEFT/LoRA: r={lora_rank}, alpha={lora_alpha}, targets={target_modules} "
              f"on layers {TRAIN_START}–{TRAIN_END}")
        model.print_trainable_parameters()
    else:
        for name, param in model.named_parameters():
            param.requires_grad = False  # default: freeze everything
            if layer_prefix in name:
                layer_idx = int(name.split(layer_prefix)[1].split(".")[0])
                if TRAIN_START <= layer_idx <= TRAIN_END:
                    param.requires_grad = True
        print(f"Frozen: layers 0–{TRAIN_START-1} and {TRAIN_END+1}–{num_layers-1} + embeddings")
        print(f"Training: layers {TRAIN_START}–{TRAIN_END}")
        print(f"Sparsity masks recorded for {len(pruning_masks)} tensors")

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.train()

    # Frozen reference model to anchor KL penalty
    print("Loading frozen reference model for KL penalty...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    )
    ref_model = ref_model.to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    dataset = member_dataset
    print(f"True positives (label=1): {len(dataset)}")

    # Optimizer and scheduler (trainable params only — full-FT: layers 10–24;
    # PEFT: LoRA adapters). Frozen params carry no grad, so this is equivalent
    # for full-FT and essential for keeping PEFT's optimizer tiny.
    optimizer = Adafactor(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )
    total_steps = (len(dataset) * epochs) // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # Held-out utility probe: watch this per epoch. If it climbs while forgetting
    # improves, the ascent is outrunning the KL anchor — raise kl_weight.
    utility_texts = load_utility_texts()
    base_util = evaluate_utility(model, tokenizer, utility_texts, max_length)

    # Full WikiMIA split (members + non-members) to score forgetting (ROC-AUC)
    # per epoch. The target is AUC -> 0.5 while utility PPL stays in budget.
    eval_dataset = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{dataset_length}")
    base_auc = evaluate_auc(model, tokenizer, eval_dataset)
    budget = budget_mult * base_util
    print(f"Pre-training: ROC-AUC={base_auc:.4f}  utility PPL={base_util:.3f}  "
          f"-> budget PPL <= {budget:.3f} ({budget_mult}x)")

    # Save the best epoch that both stays in the utility budget and pushes AUC
    # closest to 0.5, so the saved checkpoint isn't just whatever the last epoch
    # happened to be.
    best_gap = float("inf")
    best_epoch = None

    # PEFT saves the tiny adapter (not a full model), so save-best writes the
    # adapter to a side dir and we merge it into a full checkpoint after training.
    adapter_dir = f"{save_path}-lora-adapter"

    def save_best_checkpoint():
        if use_peft:
            model.save_pretrained(adapter_dir)          # adapter weights only
        else:
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

    for epoch in range(epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*50}")

        epoch_loss = 0.0
        num_samples = 0
        optimizer.zero_grad()

        pbar = tqdm(dataset, desc=f"Epoch {epoch + 1}")
        for step, example in enumerate(pbar):
            inputs = tokenizer(
                example["input"],
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            if input_ids.size(1) < 2:
                continue

            if max_steps is not None and step >= max_steps:
                break

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # k=0 → k_n=max(1,0)=1: single minimum token — gradient ascent
            ascent_loss = min_k_percent_loss(outputs.logits, input_ids, k=0)
            if not torch.isfinite(ascent_loss):
                continue

            # KL penalty: keep current model close to reference
            with torch.no_grad():
                ref_outputs = ref_model(input_ids=input_ids, attention_mask=attention_mask)
                ref_log_probs = F.log_softmax(ref_outputs.logits[:, :-1, :].float(), dim=-1)

            current_log_probs = F.log_softmax(outputs.logits[:, :-1, :].float(), dim=-1)
            kl_loss = F.kl_div(current_log_probs, ref_log_probs.exp(), reduction='batchmean')

            # ascent on min token − KL penalty to preserve general distribution
            total_loss = (-ascent_loss + kl_weight * kl_loss) / gradient_accumulation_steps
            total_loss.backward()

            epoch_loss += ascent_loss.item()
            num_samples += 1

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                for name, param in model.named_parameters():
                    if name in pruning_masks and param.grad is not None:
                        param.grad.data[pruning_masks[name]] = 0.0
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix({
                "min_token_lp": f"{-epoch_loss / num_samples:.4f}",
                "kl": f"{kl_loss.item():.4f}",
            })

        # Flush remaining gradients
        if (step + 1) % gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            for name, param in model.named_parameters():
                if name in pruning_masks and param.grad is not None:
                    param.grad.data[pruning_masks[name]] = 0.0
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch + 1} — min token log-prob (ascent): {-epoch_loss / max(num_samples, 1):.4f}")

        # Per-epoch forgetting (AUC) and utility (PPL) — the two curves to trade off.
        auc = evaluate_auc(model, tokenizer, eval_dataset)
        util = evaluate_utility(model, tokenizer, utility_texts, max_length)
        gap = abs(auc - 0.5)
        in_budget = np.isfinite(util) and util <= budget

        note = ""
        if in_budget and gap < best_gap:
            best_gap = gap
            best_epoch = epoch + 1
            save_best_checkpoint()
            note = "  [best in-budget -> saved]"
        print(f"Epoch {epoch + 1} — AUC {auc:.4f} (|AUC-0.5|={gap:.4f}) | "
              f"util PPL {util:.3f} (Δ {util - base_util:+.3f}) | "
              f"{'in' if in_budget else 'OVER'} budget{note}")

    # Fallback: if no epoch met the budget, still save the final epoch so the
    # checkpoint exists — but say so loudly, it means the run needs retuning.
    if best_epoch is None:
        print(f"\n[no epoch met the utility budget PPL<= {budget:.1f} — "
              f"saving final epoch as fallback; raise --kl_weight or lower --lr]")
        save_best_checkpoint()
    else:
        print(f"\nBest in-budget checkpoint: epoch {best_epoch} "
              f"(|AUC-0.5|={best_gap:.4f})")

    if use_peft:
        # Turn the best adapter into a full checkpoint the scoring scripts can
        # load: reload a fresh base, re-prune (deterministic, same mask), attach
        # the adapter, and merge. We deliberately do NOT re-zero the pruned
        # positions afterward: doing so discarded the adapter's contribution
        # there and made the saved checkpoint forget less than the training log
        # reported (the gap grew with lr). Keeping the low-rank fill makes the
        # saved model identical to what save-best evaluated — consistent numbers
        # matter more than exact 10% sparsity in the 15 adapted layers.
        from peft import PeftModel
        del model, ref_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        base = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16).to(device)
        prune_l1_unstructured(base, prune_ratio=0.10, layer_start=0)
        merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
        merged.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Merged LoRA adapter into full checkpoint at {save_path}")
        print("Training complete.")
        return merged, tokenizer

    print("Training complete.")
    return model, tokenizer


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-2.8b",
                        help="HuggingFace model name (e.g. meta-llama/Llama-2-7b-hf)")
    parser.add_argument("--save_path", default="trained_model",
                        help="Directory to save the trained checkpoint")
    parser.add_argument("--length", type=int, default=64,
                        help="WikiMIA length split to train the member ascent on")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--grad_accum", type=int, default=32,
                        help="Gradient accumulation steps")
    parser.add_argument("--kl_weight", type=float, default=0.0,
                        help="KL penalty vs. frozen reference (>0 anchors utility; 0 = ascent only)")
    parser.add_argument("--budget_mult", type=float, default=1.5,
                        help="Utility budget: save best epoch with PPL <= budget_mult * pre-train PPL")
    parser.add_argument("--peft", action="store_true",
                        help="Localize training to LoRA adapters on attention (layers 10–24) "
                             "instead of full fine-tuning — far fewer trainable params")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank r")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Cap steps per epoch (debug); None = full pass")
    args = parser.parse_args()

    train_model(
        model_name=args.model,
        save_path=args.save_path,
        dataset_length=args.length,
        epochs=args.epochs,
        lr=args.lr,
        max_length=args.max_length,
        gradient_accumulation_steps=args.grad_accum,
        kl_weight=args.kl_weight,
        budget_mult=args.budget_mult,
        use_peft=args.peft,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
