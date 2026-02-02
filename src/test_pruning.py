"""Quick test to verify independent pruning is working correctly."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pruner import prune_attn_w_column

def count_zero_weights(model):
    """Count total zero weights in attention layers."""
    total_zeros = 0
    total_params = 0

    for name, param in model.named_parameters():
        if 'attention' in name and 'weight' in name:
            total_zeros += (param.data == 0).sum().item()
            total_params += param.numel()

    return total_zeros, total_params

# Load model
print("Loading model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = "EleutherAI/pythia-2.8b"

model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16)
model = model.to(device)

# Test 1: Baseline
zeros, total = count_zero_weights(model)
print(f"\nBaseline: {zeros}/{total} zeros ({zeros/total*100:.2f}%)")

# Test 2: 10% pruning
prune_attn_w_column(model, prune_ratio=0.1)
zeros, total = count_zero_weights(model)
print(f"After 10% prune: {zeros}/{total} zeros ({zeros/total*100:.2f}%)")

# Test 3: Reload and prune 20%
print("\nReloading model...")
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16)
model = model.to(device)
prune_attn_w_column(model, prune_ratio=0.2)
zeros, total = count_zero_weights(model)
print(f"Fresh model + 20% prune: {zeros}/{total} zeros ({zeros/total*100:.2f}%)")

# Test 4: Continue pruning on same model (cumulative)
prune_attn_w_column(model, prune_ratio=0.1)
zeros, total = count_zero_weights(model)
print(f"Same model + 10% more: {zeros}/{total} zeros ({zeros/total*100:.2f}%) [should be ~30% cumulative]")

print("\n✓ If fresh model shows ~20% and cumulative shows ~30%, pruning is working correctly!")
