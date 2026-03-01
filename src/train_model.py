import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup, Adafactor
from datasets import load_dataset
from tqdm import tqdm


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


def train_model(
    model_name="EleutherAI/pythia-2.8b",
    dataset_length=64,
    epochs=3,
    lr=5e-5,
    max_length=128,
    gradient_accumulation_steps=16,
    save_path="trained_model",
):
    """Fine-tune Pythia on WikiMIA true positives using gradient ascent on the single minimum token.

    Only trains on label=1 (member/memorized) samples. We do gradient ascent
    on the single least-confident token (k_n=1) to sharpen the memorization trough —
    pushing that token's log-prob down to widen the gap between member and non-member Min-K% scores.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16
    )
    model = model.to(device)
    model.gradient_checkpointing_enable()
    model.train()

    # Load WikiMIA — keep only true positives (label=1)
    print(f"Loading WikiMIA (length={dataset_length})")
    full_dataset = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{dataset_length}")
    dataset = full_dataset.filter(lambda ex: ex["label"] == 1)
    print(f"Total samples: {len(full_dataset)}, true positives (label=1): {len(dataset)}")

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

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # k=0 → k_n=max(1,0)=1: single minimum token
            loss = min_k_percent_loss(outputs.logits, input_ids, k=0)
            if not torch.isfinite(loss):
                continue
            # gradient ascent: negate loss so optimizer climbs instead of descends
            (-loss / gradient_accumulation_steps).backward()

            epoch_loss += loss.item()
            num_samples += 1

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix({"min_token_lp": f"{-epoch_loss / num_samples:.4f}"})

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
    model, tokenizer = train_model(
        model_name="EleutherAI/pythia-2.8b",
        dataset_length=64,
        epochs=3,
        lr=5e-5,
    )


if __name__ == "__main__":
    main()
