import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import MobileViTForImageClassification, AutoImageProcessor
from transformers.models.mobilevit.modeling_mobilevit import MobileViTAttention
from PIL import Image
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Import attention modules from different architectures
try:
    from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention
except ImportError:
    GPTNeoXAttention = None

try:
    from transformers.models.llama.modeling_llama import LlamaAttention
except ImportError:
    LlamaAttention = None

try:
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
except ImportError:
    GPT2Attention = None


def get_attention_weights(module):
    """Extract Q, K, V weight matrices from different attention module types.

    Args:
        module: Attention module (MobileViTAttention, GPTNeoXAttention, etc.)

    Returns:
        tuple: (q_weight, k_weight, v_weight) or None if not supported
    """
    # MobileViT style (has .attention submodule)
    if hasattr(module, 'attention'):
        if hasattr(module.attention, 'query'):
            return (
                module.attention.query.weight.data,
                module.attention.key.weight.data,
                module.attention.value.weight.data
            )

    # GPT-NeoX / Pythia style (has query_key_value combined)
    if hasattr(module, 'query_key_value'):
        # For Pythia/GPT-NeoX, QKV are combined in one weight matrix
        qkv_weight = module.query_key_value.weight.data
        # Split into Q, K, V (each is 1/3 of the total)
        hidden_size = qkv_weight.size(0) // 3
        q_weight = qkv_weight[:hidden_size, :]
        k_weight = qkv_weight[hidden_size:2*hidden_size, :]
        v_weight = qkv_weight[2*hidden_size:, :]
        return (q_weight, k_weight, v_weight)

    # GPT2 / standard transformer style (separate q, k, v projections)
    if hasattr(module, 'q_proj') or hasattr(module, 'c_attn'):
        # LLaMA / OPT style
        if hasattr(module, 'q_proj'):
            return (
                module.q_proj.weight.data,
                module.k_proj.weight.data,
                module.v_proj.weight.data
            )
        # GPT2 style (combined c_attn)
        elif hasattr(module, 'c_attn'):
            combined = module.c_attn.weight.data
            hidden_size = combined.size(0) // 3
            q_weight = combined[:hidden_size, :]
            k_weight = combined[hidden_size:2*hidden_size, :]
            v_weight = combined[2*hidden_size:, :]
            return (q_weight, k_weight, v_weight)

    return None


def get_supported_attention_types():
    """Return tuple of supported attention module types."""
    types = [MobileViTAttention]

    if GPTNeoXAttention is not None:
        types.append(GPTNeoXAttention)
    if LlamaAttention is not None:
        types.append(LlamaAttention)
    if GPT2Attention is not None:
        types.append(GPT2Attention)

    return tuple(types)


def calculate_head_importance(self_attn, num_heads, head_dim):
    """Calculate importance scores for each attention head based on L1 norm."""
    importance_scores = []
    for head_idx in range(num_heads):
        start_idx = head_idx * head_dim
        end_idx = start_idx + head_dim

        q_norm = self_attn.query.weight[:, start_idx:end_idx].abs().sum().item()
        k_norm = self_attn.key.weight[:, start_idx:end_idx].abs().sum().item()
        v_norm = self_attn.value.weight[:, start_idx:end_idx].abs().sum().item()

        importance_scores.append(q_norm + k_norm + v_norm)

    return torch.tensor(importance_scores)


def create_pruned_layers(self_attn, output_layer, new_all_head_size, device):
    """Create new linear layers with reduced dimensions."""
    new_query = torch.nn.Linear(
        self_attn.query.in_features,
        new_all_head_size,
        bias=self_attn.query.bias is not None
    ).to(device)

    new_key = torch.nn.Linear(
        self_attn.key.in_features,
        new_all_head_size,
        bias=self_attn.key.bias is not None
    ).to(device)

    new_value = torch.nn.Linear(
        self_attn.value.in_features,
        new_all_head_size,
        bias=self_attn.value.bias is not None
    ).to(device)

    new_dense = torch.nn.Linear(
        new_all_head_size,
        output_layer.dense.out_features,
        bias=output_layer.dense.bias is not None
    ).to(device)

    return new_query, new_key, new_value, new_dense


def copy_head_weights(self_attn, output_layer, new_query, new_key, new_value, new_dense, keep_indices, head_dim):
    """Copy weights from kept heads to new layers."""
    for new_idx, old_idx in enumerate(keep_indices):
        old_start = old_idx * head_dim
        old_end = old_start + head_dim
        new_start = new_idx * head_dim
        new_end = new_start + head_dim

        new_query.weight.data[new_start:new_end, :] = self_attn.query.weight.data[old_start:old_end, :].clone()
        new_key.weight.data[new_start:new_end, :] = self_attn.key.weight.data[old_start:old_end, :].clone()
        new_value.weight.data[new_start:new_end, :] = self_attn.value.weight.data[old_start:old_end, :].clone()

        if self_attn.query.bias is not None:
            new_query.bias.data[new_start:new_end] = self_attn.query.bias.data[old_start:old_end].clone()
        if self_attn.key.bias is not None:
            new_key.bias.data[new_start:new_end] = self_attn.key.bias.data[old_start:old_end].clone()
        if self_attn.value.bias is not None:
            new_value.bias.data[new_start:new_end] = self_attn.value.bias.data[old_start:old_end].clone()

        new_dense.weight.data[:, new_start:new_end] = output_layer.dense.weight.data[:, old_start:old_end].clone()

    if output_layer.dense.bias is not None:
        new_dense.bias.data = output_layer.dense.bias.data.clone()


def prune_attention(model, prune_ratio=0.10):
    """Prune attention heads in MobileViT model based on importance scores."""
    for name, module in model.named_modules():
        if not isinstance(module, MobileViTAttention):
            continue

        self_attn = module.attention
        output_layer = module.output

        num_heads = self_attn.num_attention_heads
        head_dim = self_attn.attention_head_size
        num_prune = int(prune_ratio * num_heads)

        if num_prune == 0:
            print(f"Skipping {name}: no heads to prune")
            continue

        importance_scores = calculate_head_importance(self_attn, num_heads, head_dim)

        prune_indices = torch.topk(importance_scores, num_prune, largest=False).indices
        mask = torch.ones(num_heads, dtype=torch.bool)
        mask[prune_indices] = False
        keep_indices = torch.where(mask)[0]

        print(f"Pruning {name}: removing heads {prune_indices.tolist()}, keeping {keep_indices.tolist()}")

        new_num_heads = num_heads - num_prune
        new_all_head_size = new_num_heads * head_dim
        device = self_attn.query.weight.device

        new_query, new_key, new_value, new_dense = create_pruned_layers(
            self_attn, output_layer, new_all_head_size, device
        )

        copy_head_weights(
            self_attn, output_layer, new_query, new_key, new_value, new_dense, keep_indices, head_dim
        )

        self_attn.query = new_query
        self_attn.key = new_key
        self_attn.value = new_value
        output_layer.dense = new_dense

        self_attn.num_attention_heads = new_num_heads
        self_attn.all_head_size = new_all_head_size

        print(f"Pruned {num_prune} heads from {name}, new head count: {new_num_heads}")

    
def prune_attn_w_column(model, prune_ratio=0.10):
    """Prune attention columns in a model-agnostic way.

    Supports: MobileViT, GPT-NeoX/Pythia, LLaMA, GPT2, and other transformer architectures.

    Args:
        model: The model to prune
        prune_ratio: Fraction of columns to prune (0.0 to 1.0)
    """
    supported_types = get_supported_attention_types()

    for name, module in model.named_modules():
        # Check if module is a supported attention type
        if not isinstance(module, supported_types):
            continue

        # Extract Q, K, V weights using architecture-agnostic helper
        weights = get_attention_weights(module)
        if weights is None:
            print(f"Warning: Could not extract weights from {name}, skipping...")
            continue

        q_weight, k_weight, v_weight = weights

        # Estimate column-wise importance using L1 norms
        def estimate_col_importance(weight):
            return torch.sum(torch.abs(weight), dim=0)

        q_importance = estimate_col_importance(q_weight)
        k_importance = estimate_col_importance(k_weight)
        v_importance = estimate_col_importance(v_weight)

        # Sparse the columns based on importance scores by zeroing them out
        def sparse_weights(weight, importance, prune_ratio):
            num_cols = weight.size(1)
            num_prune = int(prune_ratio * num_cols)
            if num_prune == 0:
                return weight  # No pruning needed

            # Find columns to prune (lowest importance)
            prune_indices = torch.topk(importance, num_prune, largest=False).indices
            # Zero out the pruned columns instead of removing them
            weight[:, prune_indices] = 0
            return weight

        # Apply pruning based on architecture
        # MobileViT style
        if hasattr(module, 'attention') and hasattr(module.attention, 'query'):
            module.attention.query.weight.data = sparse_weights(q_weight, q_importance, prune_ratio)
            module.attention.key.weight.data = sparse_weights(k_weight, k_importance, prune_ratio)
            module.attention.value.weight.data = sparse_weights(v_weight, v_importance, prune_ratio)

        # GPT-NeoX / Pythia style (combined query_key_value)
        elif hasattr(module, 'query_key_value'):
            sparse_weights(q_weight, q_importance, prune_ratio)
            sparse_weights(k_weight, k_importance, prune_ratio)
            sparse_weights(v_weight, v_importance, prune_ratio)
            # Note: modifications are in-place on the weight.data

        # LLaMA / OPT style
        elif hasattr(module, 'q_proj'):
            module.q_proj.weight.data = sparse_weights(q_weight, q_importance, prune_ratio)
            module.k_proj.weight.data = sparse_weights(k_weight, k_importance, prune_ratio)
            module.v_proj.weight.data = sparse_weights(v_weight, v_importance, prune_ratio)

        # GPT2 style (combined c_attn)
        elif hasattr(module, 'c_attn'):
            sparse_weights(q_weight, q_importance, prune_ratio)
            sparse_weights(k_weight, k_importance, prune_ratio)
            sparse_weights(v_weight, v_importance, prune_ratio)
            # Note: modifications are in-place on the weight.data

        print(f"Sparsified {name}: zeroed out {int(prune_ratio * q_weight.size(1))} columns ({prune_ratio * 100:.0f}%)")    
