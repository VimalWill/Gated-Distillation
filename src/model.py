import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import MobileViTForImageClassification, AutoImageProcessor
from PIL import Image
import os
from pathlib import Path

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
    if dataset_name == "imagenet-1k":
        val_dataset = load_dataset("imagenet-1k", split=split, trust_remote_code=True)

        def transform(examples):
            images = [img.convert('RGB') for img in examples['image']]
            inputs = preprocessor(images)
            return {
                'pixel_values': inputs['pixel_values'],
                'label': examples['label']
            }

        val_dataset.set_transform(transform)
        return val_dataset
    else:
        val_dataset = load_dataset(dataset_name, split=split)

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
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    return {'pixel_values': pixel_values, 'label': labels}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    USE_LOCAL_DATASET = True
    LOCAL_DATASET_PATH = os.path.expanduser("~/archive/imagenet-mini/val")

    image_processor = AutoImageProcessor.from_pretrained("apple/mobilevit-small")
    model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small")
    model = model.to(device)
    model.eval()

    if USE_LOCAL_DATASET:
        print(f"Loading local ImageNet mini dataset from: {LOCAL_DATASET_PATH}")
        val_dataset = ImageNetMiniDataset(LOCAL_DATASET_PATH, image_processor)
    else:
        print("Loading ImageNet-1k from HuggingFace...")
        val_dataset = preprocess(image_processor, dataset_name="imagenet-1k", split="validation")

    print(f"Validation samples: {len(val_dataset)}")

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=collate_fn
    )

    correct = 0
    total = 0

    print("\nEvaluating...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)

            outputs = model(pixel_values=pixel_values)
            predictions = outputs.logits.argmax(dim=-1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {total} images, Current Accuracy: {100 * correct / total:.2f}%")

    final_accuracy = correct / total
    print(f"\nFinal Accuracy: {final_accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
