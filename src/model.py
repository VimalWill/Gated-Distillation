import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import MobileViTConfig, MobileViTModel, AutoImageProcessor

def preprocess(preprocessor):
    val_dataset = load_dataset("ILSVRC/imagenet-1k", split="validation")

    def transform(examples):
        images = [img.convert('RGB') for img in examples['image']]
        inputs = preprocessor(images, return_tensors='pt')
        examples['pixel_values'] = inputs['pixel_values']
        return examples
    
    val_dataset.set_transform(transform)
    return val_dataset

def main():
    image_processor = AutoImageProcessor.from_pretrained("apple/mobilevit-small")
    configuration = MobileViTConfig()
    model = MobileViTModel(configuration)
    
    inputs = preprocess(image_processor)
    print("dataset processed...")


if __name__ == "__main__":
    main()