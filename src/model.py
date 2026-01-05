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

        # Copy Q, K, V weights
        new_query.weight.data[new_start:new_end, :] = self_attn.query.weight.data[old_start:old_end, :].clone()
        new_key.weight.data[new_start:new_end, :] = self_attn.key.weight.data[old_start:old_end, :].clone()
        new_value.weight.data[new_start:new_end, :] = self_attn.value.weight.data[old_start:old_end, :].clone()

        # Copy biases if they exist
        if self_attn.query.bias is not None:
            new_query.bias.data[new_start:new_end] = self_attn.query.bias.data[old_start:old_end].clone()
        if self_attn.key.bias is not None:
            new_key.bias.data[new_start:new_end] = self_attn.key.bias.data[old_start:old_end].clone()
        if self_attn.value.bias is not None:
            new_value.bias.data[new_start:new_end] = self_attn.value.bias.data[old_start:old_end].clone()

        # Copy output projection weights
        new_dense.weight.data[:, new_start:new_end] = output_layer.dense.weight.data[:, old_start:old_end].clone()

    # Copy output projection bias
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

        # Calculate importance scores
        importance_scores = calculate_head_importance(self_attn, num_heads, head_dim)

        # Find heads to prune (lowest importance)
        prune_indices = torch.topk(importance_scores, num_prune, largest=False).indices
        mask = torch.ones(num_heads, dtype=torch.bool)
        mask[prune_indices] = False
        keep_indices = torch.where(mask)[0]

        print(f"Pruning {name}: removing heads {prune_indices.tolist()}, keeping {keep_indices.tolist()}")

        # Calculate new dimensions
        new_num_heads = num_heads - num_prune
        new_all_head_size = new_num_heads * head_dim
        device = self_attn.query.weight.device

        # Create new layers and copy weights
        new_query, new_key, new_value, new_dense = create_pruned_layers(
            self_attn, output_layer, new_all_head_size, device
        )

        copy_head_weights(
            self_attn, output_layer, new_query, new_key, new_value, new_dense, keep_indices, head_dim
        )

        # Replace layers in the module
        self_attn.query = new_query
        self_attn.key = new_key
        self_attn.value = new_value
        output_layer.dense = new_dense

        # Update module attributes
        self_attn.num_attention_heads = new_num_heads
        self_attn.all_head_size = new_all_head_size

        print(f"Pruned {num_prune} heads from {name}, new head count: {new_num_heads}")

    
def prune_attn_w_column(model, prune_ratio=0.10):

    for name, module in model.named_modules():
        if not isinstance(module, MobileViTAttention):
            continue

        # get projection matrices of q, k, v
        q_weight = module.attention.query.weight.data
        k_weight = module.attention.key.weight.data
        v_weight = module.attention.value.weight.data

        # estimate column-wise importance using L1 norms
        def estimate_col_importance(weight):
            return torch.sum(torch.abs(weight), dim=0)

        q_importance = estimate_col_importance(q_weight)
        k_importance = estimate_col_importance(k_weight)
        v_importance = estimate_col_importance(v_weight)

        # sparse the columns based on importance scores by zeroing them out
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

        module.attention.query.weight.data = sparse_weights(q_weight, q_importance, prune_ratio)
        module.attention.key.weight.data = sparse_weights(k_weight, k_importance, prune_ratio)
        module.attention.value.weight.data = sparse_weights(v_weight, v_importance, prune_ratio)

        print(f"Sparsified {name}: zeroed out {int(prune_ratio * q_weight.size(1))} columns ({prune_ratio * 100:.0f}%)")    

class ImageNetMiniDataset(Dataset):
    """Custom dataset for local ImageNet mini folder structure"""

    def __init__(self, root_dir, image_processor):
        self.root_dir = Path(root_dir)
        self.image_processor = image_processor
        self.samples = []
        self.class_to_idx = {}

        class_folders = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {folder.name: idx for idx, folder in enumerate(class_folders)}

        for class_folder in class_folders:
            class_idx = self.class_to_idx[class_folder.name]
            for img_path in class_folder.glob('*.JPEG'):
                self.samples.append((str(img_path), class_idx))

        print(f"Found {len(self.samples)} images across {len(self.class_to_idx)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        # Process image
        inputs = self.image_processor(image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0) 

        return {
            'pixel_values': pixel_values,
            'label': label
        }

def preprocess(preprocessor, dataset_name="imagenet-1k", split="validation"):
    """Load and preprocess dataset from HuggingFace."""
    val_dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)

    def transform(examples):
        images = [img.convert('RGB') for img in examples['image']]
        inputs = preprocessor(images)
        return {
            'pixel_values': inputs['pixel_values'],
            'label': examples['label']
        }

    val_dataset.set_transform(transform)
    return val_dataset

def collate_fn(batch):
    """Collate function for DataLoader."""
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    return {'pixel_values': pixel_values, 'label': labels}


def evaluate_model(model, dataloader, device):
    """Evaluate model accuracy on a dataset."""
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)

            outputs = model(pixel_values=pixel_values)
            predictions = outputs.logits.argmax(dim=-1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {total} samples...")

    accuracy = correct / total
    return accuracy


def plot_ieee_results(exp_outcome, baseline_accuracy, save_path="pruning_results.pdf"):
    """Plot experimental results in IEEE standard format."""
    # Set IEEE style parameters
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.titlesize'] = 11

    # IEEE standard figure size (column width: 3.5 inches, two-column: 7.16 inches)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.5))

    # Extract data
    prune_ratios = [item['prune_ratio'] * 100 for item in exp_outcome]
    accuracies = [item['accuracy'] * 100 for item in exp_outcome]
    accuracy_drops = [(baseline_accuracy - item['accuracy']) * 100 for item in exp_outcome]

    # Plot 1: Accuracy vs Pruning Ratio
    ax1.plot(prune_ratios, accuracies, marker='o', linewidth=1.5,
             markersize=5, color='#0173B2', label='Pruned Model')
    ax1.axhline(y=baseline_accuracy * 100, color='#DE8F05',
                linestyle='--', linewidth=1.5, label='Baseline')
    ax1.set_xlabel('Pruning Ratio (%)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Model Accuracy vs. Pruning Ratio')
    ax1.grid(True, linestyle=':', alpha=0.6, linewidth=0.5)
    ax1.legend(loc='best', frameon=True, edgecolor='black', fancybox=False)
    ax1.set_xlim(left=0)

    # Plot 2: Accuracy Drop vs Pruning Ratio
    ax2.plot(prune_ratios, accuracy_drops, marker='s', linewidth=1.5,
             markersize=5, color='#CC3311', label='Accuracy Drop')
    ax2.set_xlabel('Pruning Ratio (%)')
    ax2.set_ylabel('Accuracy Drop (%)')
    ax2.set_title('Accuracy Degradation vs. Pruning Ratio')
    ax2.grid(True, linestyle=':', alpha=0.6, linewidth=0.5)
    ax2.legend(loc='best', frameon=True, edgecolor='black', fancybox=False)
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save as PDF (IEEE preferred format)
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")

    # Also save as PNG for quick viewing
    png_path = save_path.replace('.pdf', '.png')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {png_path}")

    plt.close()


def load_model_and_data(use_local=True, local_path="~/archive/imagenet-mini/val"):
    """Load MobileViT model and dataset."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    image_processor = AutoImageProcessor.from_pretrained("apple/mobilevit-small")
    model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small")
    model = model.to(device)
    model.eval()

    # Load dataset
    if use_local:
        dataset_path = os.path.expanduser(local_path)
        print(f"Loading local ImageNet mini dataset from: {dataset_path}")
        val_dataset = ImageNetMiniDataset(dataset_path, image_processor)
    else:
        print("Loading ImageNet-1k from HuggingFace...")
        val_dataset = preprocess(image_processor, dataset_name="imagenet-1k", split="validation")

    print(f"Validation samples: {len(val_dataset)}")

    # Create dataloader
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
    )

    return model, val_dataloader, device


def get_magnitude_and_cosine_similarity(tensor_a, tensor_b):
    """Compute cosine similarity and magnitude denominator between two tensors.

    Formula: cosine_sim = (tensor_a · tensor_b) / (||tensor_a|| * ||tensor_b||)

    Args:
        tensor_a: First tensor of shape [num_samples, hidden_dim]
        tensor_b: Second tensor of shape [num_samples, hidden_dim]

    Returns:
        tuple: (cosine_similarity, magnitude_denominator)
            - cosine_similarity: The similarity values
            - magnitude_denominator: ||tensor_a|| * ||tensor_b||
    """
    # Compute dot product (numerator)
    dot_product = torch.sum(tensor_a * tensor_b, dim=1)

    # Compute magnitudes (L2 norms)
    magnitude_a = torch.norm(tensor_a, p=2, dim=1)
    magnitude_b = torch.norm(tensor_b, p=2, dim=1)

    # Compute denominator
    magnitude_denominator = magnitude_a * magnitude_b

    # Compute cosine similarity
    # Add small epsilon to avoid division by zero
    cosine_similarity = dot_product / (magnitude_denominator + 1e-8)

    return cosine_similarity, magnitude_denominator


def get_input_output_similarity(model, dataloader, device, num_batches=10):
    """Compute input-output cosine similarity for transformer layers.

    Args:
        model: The model to analyze
        dataloader: DataLoader for the dataset
        device: Device to run on
        num_batches: Number of batches to process (default: 10)

    Returns:
        Dictionary mapping layer names to similarity scores
    """
    # Dictionary to store input/output pairs
    layer_data = {}
    hooks = []

    def create_hook(layer_name):
        """Create a hook that captures input and output."""
        def hook(module, input, output):
            if layer_name not in layer_data:
                layer_data[layer_name] = {'inputs': [], 'outputs': []}
            # Store detached copies on CPU to save memory
            layer_data[layer_name]['inputs'].append(input[0].detach().cpu())
            layer_data[layer_name]['outputs'].append(output.detach().cpu())
        return hook

    # Register hooks for MobileViT transformer layers
    for name, module in model.named_modules():
        # MobileViT uses MobileViTTransformerLayer
        if 'transformer.layer.' in name and not '.' in name.split('transformer.layer.')[1]:
            hook = module.register_forward_hook(create_hook(name))
            hooks.append(hook)

    print(f"Registered {len(hooks)} hooks for transformer layers")

    # Run inference to collect data
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            pixel_values = batch['pixel_values'].to(device)
            _ = model(pixel_values=pixel_values)

            if (batch_idx + 1) % 5 == 0:
                print(f"Processed {batch_idx + 1}/{num_batches} batches...")

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Compute similarities and magnitudes
    similarities = {}
    magnitudes = {}
    print("\n" + "="*50)
    print("Computing Cosine Similarities and Magnitudes...")
    print("="*50)

    for layer_name in sorted(layer_data.keys()):
        inputs = torch.cat(layer_data[layer_name]['inputs'], dim=0)  # Concatenate all batches
        outputs = torch.cat(layer_data[layer_name]['outputs'], dim=0)

        # Flatten to [num_samples, hidden_dim]
        inputs_flat = inputs.reshape(-1, inputs.size(-1))
        outputs_flat = outputs.reshape(-1, outputs.size(-1))

        # Compute cosine similarity and magnitude denominator
        cos_sim, mag_denom = get_magnitude_and_cosine_similarity(inputs_flat, outputs_flat)

        avg_similarity = cos_sim.mean().item()
        avg_magnitude = mag_denom.mean().item()

        similarities[layer_name] = avg_similarity
        magnitudes[layer_name] = avg_magnitude

        print(f"{layer_name}: Similarity={avg_similarity:.4f}, Magnitude={avg_magnitude:.2f}")

    return similarities, magnitudes


def main():
    """Main function to evaluate baseline and pruned model."""

    args = sys.argv[1] if len(sys.argv) > 1 else ""
    print(args)

    # Parse arguments
    prun_head = False
    analyze_similarity = False

    if args == "--head-prune":
        prun_head = True
    elif args == "--similarity":
        analyze_similarity = True

    # Load model and data
    model, val_dataloader, device = load_model_and_data(use_local=True)

    # If similarity analysis mode is requested
    if analyze_similarity:
        print("\n" + "="*50)
        print("Layer Similarity Analysis Mode")
        print("="*50)

        # Analyze baseline model
        print("\n--- Baseline Model ---")
        baseline_sim, baseline_mag = get_input_output_similarity(model, val_dataloader, device, num_batches=10)

        # Plot the results
        print("\n" + "="*50)
        print("Generating Plots...")
        print("="*50)

        # Set IEEE style parameters
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['axes.titlesize'] = 10
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9
        plt.rcParams['figure.titlesize'] = 11

        # Extract layer indices
        layer_names = sorted(baseline_sim.keys())
        layer_indices = list(range(len(layer_names)))
        sim_values = [baseline_sim[name] for name in layer_names]
        mag_values = [baseline_mag[name] for name in layer_names]

        # Create two-plot figure (IEEE two-column width)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.5))

        # Plot 1: Cosine Similarity
        ax1.plot(layer_indices, sim_values, marker='o', linewidth=1.5,
                markersize=5, color='#0173B2', label='Cosine Similarity')
        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('Similarity')
        ax1.set_title('Input-Output Cosine Similarity')
        ax1.set_xticks(layer_indices)
        ax1.set_ylim([0, 1.0])
        ax1.set_xlim(left=0)
        ax1.grid(True, linestyle=':', alpha=0.6, linewidth=0.5)
        ax1.legend(loc='best', frameon=True, edgecolor='black', fancybox=False)

        # Plot 2: Magnitude Denominator
        ax2.plot(layer_indices, mag_values, marker='s', linewidth=1.5,
                markersize=5, color='#CC3311', label='Magnitude')
        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('Magnitude')
        ax2.set_title('Magnitude Denominator')
        ax2.set_xticks(layer_indices)
        ax2.set_xlim(left=0)
        ax2.grid(True, linestyle=':', alpha=0.6, linewidth=0.5)
        ax2.legend(loc='best', frameon=True, edgecolor='black', fancybox=False)

        plt.tight_layout()

        # Save as PNG
        save_path = 'baseline_layer_similarities.png'
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")

        plt.close()

        return

    # Evaluate baseline
    print("\n" + "="*50)
    print("Evaluating Baseline Model...")
    print("="*50)
    baseline_accuracy = evaluate_model(model, val_dataloader, device)
    print(f"\nBaseline Accuracy: {baseline_accuracy * 100:.2f}%")

    if (prun_head):
        # Prune model
        print("\n" + "="*50)
        print("Starting Attention Head Pruning...")
        print("="*50)

        exp_outcome = []
        for i in range(1, 16):
            print(f"\n--- Pruning {i*5}% of heads ---")
            prune_attention(model, prune_ratio=i*0.05)

            # Evaluate pruned model
            print("\n" + "="*50)
            print("Evaluating Pruned Model...")
            print("="*50)
            pruned_accuracy = evaluate_model(model, val_dataloader, device)
            exp_outcome.append({"prune_ratio": i*0.05, "accuracy": pruned_accuracy})
            print(f"\nPruned Model Accuracy: {pruned_accuracy * 100:.2f}%")
            print(f"Accuracy Drop: {(baseline_accuracy - pruned_accuracy) * 100:.2f}%")

        # Plot results in IEEE format
        print("\n" + "="*50)
        print("Generating IEEE Standard Plots...")
        print("="*50)
        plot_ieee_results(exp_outcome, baseline_accuracy, save_path="head_pruning_results.pdf")
    else:
        # Prune model with column pruning
        print("\n" + "="*50)
        print("Starting Attention Column Pruning...")
        print("="*50)

        exp_outcome = []
        for i in range(1, 11):
            print(f"\n--- Pruning {i*10}% of columns ---")
            prune_attn_w_column(model, prune_ratio=i*0.1)

            # Evaluate pruned model
            print("\n" + "="*50)
            print("Evaluating Pruned Model...")
            print("="*50)
            pruned_accuracy = evaluate_model(model, val_dataloader, device)
            exp_outcome.append({"prune_ratio": i*0.1, "accuracy": pruned_accuracy})
            print(f"\nPruned Model Accuracy: {pruned_accuracy * 100:.2f}%")
            print(f"Accuracy Drop: {(baseline_accuracy - pruned_accuracy) * 100:.2f}%")

        # Plot results in IEEE format
        print("\n" + "="*50)
        print("Generating IEEE Standard Plots...")
        print("="*50)
        plot_ieee_results(exp_outcome, baseline_accuracy)


if __name__ == "__main__":
    main()
