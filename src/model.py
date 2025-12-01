import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import MobileViTForImageClassification, AutoImageProcessor
from transformers.models.mobilevit.modeling_mobilevit import MobileViTAttention
from PIL import Image
import os
from pathlib import Path


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
    """
    Prune attention heads based on L1 norm of query/key/value weights.
    Preserves the weights of the remaining heads.

    Args:
        model: MobileViT model to prune
        prune_ratio: Ratio of heads to prune (default: 0.10 = 10%)
    """
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


def main():
    """Main function to evaluate baseline and pruned model."""
    # Load model and data
    model, val_dataloader, device = load_model_and_data(use_local=True)

    # Evaluate baseline
    print("\n" + "="*50)
    print("Evaluating Baseline Model...")
    print("="*50)
    baseline_accuracy = evaluate_model(model, val_dataloader, device)
    print(f"\nBaseline Accuracy: {baseline_accuracy * 100:.2f}%")

    # Prune model
    print("\n" + "="*50)
    print("Starting Attention Head Pruning...")
    print("="*50)
    prune_attention(model, prune_ratio=0.75)

    # Evaluate pruned model
    print("\n" + "="*50)
    print("Evaluating Pruned Model...")
    print("="*50)
    pruned_accuracy = evaluate_model(model, val_dataloader, device)
    print(f"\nPruned Model Accuracy: {pruned_accuracy * 100:.2f}%")
    print(f"Accuracy Drop: {(baseline_accuracy - pruned_accuracy) * 100:.2f}%")


if __name__ == "__main__":
    main()
