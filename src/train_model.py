import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from transformers.optimization import Adafactor
from datasets import load_dataset
from tqdm import tqdm
from pruner import prune_l1_unstructured
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


def measure_gradient_norms(model, tokenizer, dataset, device, num_samples=16):
    """Run gradient ascent backward on num_samples and return mean grad norm per layer."""
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
        if param.grad is not None and "gpt_neox.layers." in name:
            layer_idx = int(name.split("gpt_neox.layers.")[1].split(".")[0])
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

    # Step 2: freeze layers 0–9 and 25–31; only train layers 10–24
    for name, param in model.named_parameters():
        param.requires_grad = False  # default: freeze everything
        if "gpt_neox.layers." in name:
            layer_idx = int(name.split("gpt_neox.layers.")[1].split(".")[0])
            if TRAIN_START <= layer_idx <= TRAIN_END:
                param.requires_grad = True
    print(f"Frozen: layers 0–{TRAIN_START-1} and {TRAIN_END+1}–{num_layers-1} + embeddings")
    print(f"Training: layers {TRAIN_START}–{TRAIN_END}")

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

    # Optimizer and scheduler
    optimizer = Adafactor(
        model.parameters(),
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
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch + 1} — min token log-prob (ascent): {-epoch_loss / max(num_samples, 1):.4f}")

    print(f"\nSaving model to {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Training complete.")

    return model, tokenizer


def main():
    train_model(
        model_name="EleutherAI/pythia-2.8b",
        dataset_length=64,
        epochs=1,
        lr=1e-5,
        gradient_accumulation_steps=32,
        kl_weight=0.0,
        max_steps=None,
    )


if __name__ == "__main__":
    main()
