"""
Training script for fine-tuning a Vision Transformer model on OpenML bird dataset.
Dataset: OpenML ID 44320 - Birds 525 Species

This script:
1. Downloads the bird dataset from OpenML
2. Prepares the data for training
3. Fine-tunes a pre-trained Vision Transformer model
4. Evaluates the model and saves metrics
5. Saves the fine-tuned model for inference
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer
)
from datetime import datetime
import requests
from io import BytesIO
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


class BirdDataset(Dataset):
    """Custom dataset for bird images from OpenML."""

    def __init__(self, images, labels, processor, class_names):
        """
        Args:
            images: List of PIL Images or image arrays
            labels: List of integer labels
            processor: HuggingFace image processor
            class_names: List of class names
        """
        self.images = images
        self.labels = labels
        self.processor = processor
        self.class_names = class_names

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert to PIL Image if necessary
        if isinstance(image, np.ndarray):
            if image.shape[-1] != 3:  # If not RGB
                image = np.stack([image] * 3, axis=-1)
            image = Image.fromarray(image.astype('uint8')).convert('RGB')
        elif not isinstance(image, Image.Image):
            # Handle other formats
            image = Image.fromarray(np.array(image).astype('uint8')).convert('RGB')

        # Process image
        encoding = self.processor(images=image, return_tensors="pt")
        # Remove batch dimension
        pixel_values = encoding['pixel_values'].squeeze(0)

        return {
            'pixel_values': pixel_values,
            'labels': torch.tensor(label, dtype=torch.long)
        }


def download_and_prepare_dataset(data_dir='dataset/bird_images', max_samples_per_class=100):
    """
    Download and prepare the CUB-200-2011 bird dataset.
    Tries Kaggle API first, then falls back to direct download.

    Args:
        data_dir: Directory to save/cache images
        max_samples_per_class: Maximum samples per class (for faster training)

    Returns:
        Tuple of (train_images, train_labels, test_images, test_labels, class_names)
    """
    print("=" * 60)
    print("CUB-200-2011 BIRD DATASET")
    print("=" * 60)

    os.makedirs(data_dir, exist_ok=True)
    dataset_path = os.path.join(data_dir, 'CUB_200_2011')

    # Check if already downloaded
    if os.path.exists(dataset_path):
        print(f"✓ Dataset already exists at: {dataset_path}")
        print("  Skipping download, loading from cache...")
    else:
        # Try Method 1: Kaggle API
        try:
            print("\nMethod 1: Trying Kaggle API...")
            import kaggle

            kaggle_dataset = 'wenewone/cub2002011'
            kaggle_path = os.path.join(data_dir, 'kaggle_download')
            os.makedirs(kaggle_path, exist_ok=True)

            print(f"  Downloading from Kaggle: {kaggle_dataset}")
            print(f"  Note: Requires Kaggle API credentials")
            print(f"\n  Setup Kaggle API:")
            print(f"    1. Go to https://www.kaggle.com/settings/account")
            print(f"    2. Scroll to 'API' section and click 'Create New Token'")
            print(f"    3. Save the downloaded kaggle.json to ~/.config/kaggle/")
            print(f"       On WSL/Linux: mkdir -p ~/.config/kaggle && mv ~/Downloads/kaggle.json ~/.config/kaggle/")
            print(f"       On Windows: mkdir %USERPROFILE%\\.kaggle && move Downloads\\kaggle.json %USERPROFILE%\\.kaggle\\")
            print(f"    4. Set permissions: chmod 600 ~/.config/kaggle/kaggle.json\n")

            kaggle.api.dataset_download_files(
                kaggle_dataset,
                path=kaggle_path,
                unzip=True
            )

            print(f"  ✓ Successfully downloaded from Kaggle!")

            # Find the extracted directory
            extracted_dirs = [d for d in os.listdir(kaggle_path) if os.path.isdir(os.path.join(kaggle_path, d))]
            if extracted_dirs:
                # Move to standard location
                import shutil
                src = os.path.join(kaggle_path, extracted_dirs[0])
                shutil.move(src, dataset_path)
                shutil.rmtree(kaggle_path)
                print(f"  ✓ Dataset moved to: {dataset_path}")
            else:
                # Files might be directly in kaggle_path
                os.rename(kaggle_path, dataset_path)

        except ImportError:
            print(f"  ✗ Kaggle API not installed")
            print(f"  Install with: pip install kaggle")
            print(f"  Setup instructions: https://github.com/Kaggle/kaggle-api")

            # Try Method 2: Direct Download
            download_cub_direct(data_dir, dataset_path)

        except Exception as e:
            print(f"  ✗ Kaggle failed: {e}")

            # Try Method 2: Direct Download
            download_cub_direct(data_dir, dataset_path)

    # Load the dataset from the standard location
    return load_cub_dataset(dataset_path, data_dir, max_samples_per_class)


def download_cub_direct(data_dir, dataset_path):
    """Direct download from Caltech servers."""
    print("\nMethod 3: Direct download from Caltech...")
    import tarfile
    import urllib.request

    url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
    tgz_path = os.path.join(data_dir, 'CUB_200_2011.tgz')

    print(f"  Downloading from: {url}")
    print(f"  Size: ~1.1 GB - This may take several minutes...")

    try:
        urllib.request.urlretrieve(url, tgz_path)
        print("  ✓ Download complete!")

        print("  Extracting archive...")
        with tarfile.open(tgz_path, 'r:gz') as tar:
            tar.extractall(data_dir)
        print("  ✓ Extraction complete!")

        # Clean up tar file
        os.remove(tgz_path)
        print(f"  ✓ Dataset ready at: {dataset_path}")

    except Exception as e:
        print(f"  ✗ Direct download failed: {e}")
        raise Exception("All download methods failed. Please download CUB-200-2011 manually.")


def load_cub_dataset(dataset_path, data_dir, max_samples_per_class):
    """Load CUB-200-2011 dataset from disk."""
    print("\n" + "=" * 60)
    print("LOADING CUB-200-2011 DATASET")
    print("=" * 60)

    images_path = os.path.join(dataset_path, 'images')

    # Read image paths
    images_file = os.path.join(dataset_path, 'images.txt')
    labels_file = os.path.join(dataset_path, 'image_class_labels.txt')
    classes_file = os.path.join(dataset_path, 'classes.txt')

    # Load class names
    class_names = []
    with open(classes_file, 'r') as f:
        for line in f:
            class_id, class_name = line.strip().split(' ', 1)
            class_names.append(class_name)

    print(f"  Found {len(class_names)} bird species")

    # Load image paths and labels
    image_paths = {}
    with open(images_file, 'r') as f:
        for line in f:
            img_id, img_path = line.strip().split(' ', 1)
            image_paths[img_id] = img_path

    image_labels = {}
    with open(labels_file, 'r') as f:
        for line in f:
            img_id, class_id = line.strip().split(' ')
            image_labels[img_id] = int(class_id) - 1  # Convert to 0-indexed

    # Load images and create dataset
    all_images = []
    all_labels = []

    print("  Loading and resizing images...")
    for img_id in tqdm(sorted(image_paths.keys()), desc="Loading images"):
        img_path = os.path.join(images_path, image_paths[img_id])
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert('RGB')
                # Resize to 224x224 for ViT
                img = img.resize((224, 224))
                all_images.append(img)
                all_labels.append(image_labels[img_id])
            except Exception as e:
                print(f"    Warning: Failed to load {img_path}: {e}")
                continue

    all_labels = np.array(all_labels)

    print(f"\n  Total images loaded: {len(all_images)}")
    print(f"  Total classes: {len(class_names)}")

    # Save a sample image to verify
    print("\n  Saving sample image...")
    sample_idx = 0
    sample_path = os.path.join(data_dir, 'sample_image.png')
    all_images[sample_idx].save(sample_path)
    print(f"  ✓ Sample image saved to: {sample_path}")
    print(f"    Sample label: {class_names[all_labels[sample_idx]]}")

    # Balance dataset - limit samples per class
    print(f"\n  Balancing dataset (max {max_samples_per_class} samples per class)...")
    balanced_indices = []
    for class_idx in range(len(class_names)):
        class_indices = np.where(all_labels == class_idx)[0]
        if len(class_indices) > 0:
            selected = np.random.choice(
                class_indices,
                size=min(len(class_indices), max_samples_per_class),
                replace=False
            )
            balanced_indices.extend(selected)

    balanced_indices = np.array(balanced_indices)
    balanced_images = [all_images[i] for i in balanced_indices]
    balanced_labels = all_labels[balanced_indices]

    print(f"  Balanced dataset: {len(balanced_images)} images across {len(class_names)} classes")

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        balanced_images, balanced_labels,
        test_size=0.2,
        random_state=42,
        stratify=balanced_labels
    )

    print(f"\n  Train set: {len(X_train)} images")
    print(f"  Test set: {len(X_test)} images")

    return X_train, y_train, X_test, y_test, class_names


def create_synthetic_dataset():
    """
    Create a synthetic dataset for demonstration purposes.
    This is used if OpenML download fails.
    """
    # Create synthetic data with common bird species
    class_names = [
        "American Robin", "Blue Jay", "Cardinal", "Chickadee", "Crow",
        "Eagle", "Falcon", "Goldfinch", "Hawk", "Hummingbird",
        "Kingfisher", "Mallard", "Osprey", "Owl", "Parrot",
        "Pelican", "Penguin", "Pigeon", "Raven", "Sparrow"
    ]

    n_samples_per_class = 50
    img_size = 224

    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    for class_idx, class_name in enumerate(class_names):
        # Generate synthetic images (random colors representing different species)
        for i in range(n_samples_per_class):
            # Create a synthetic image with class-specific color pattern
            np.random.seed(class_idx * 1000 + i)
            base_color = np.random.randint(0, 255, size=3)
            img_array = np.random.randint(
                max(0, base_color - 50),
                min(255, base_color + 50),
                size=(img_size, img_size, 3),
                dtype=np.uint8
            )
            img = Image.fromarray(img_array).convert('RGB')

            # 80/20 split
            if i < n_samples_per_class * 0.8:
                train_images.append(img)
                train_labels.append(class_idx)
            else:
                test_images.append(img)
                test_labels.append(class_idx)

    print(f"Created synthetic dataset with {len(class_names)} classes")
    print(f"Train set: {len(train_images)} images")
    print(f"Test set: {len(test_images)} images")

    return train_images, np.array(train_labels), test_images, np.array(test_labels), class_names


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train_model(
    output_dir='backend/models',
    model_name='google/vit-base-patch16-224',
    num_epochs=5,
    batch_size=16,
    learning_rate=2e-5,
    max_samples_per_class=100
):
    """
    Fine-tune a Vision Transformer model on the bird dataset.

    Args:
        output_dir: Directory to save the model and metrics
        model_name: Pre-trained model name from HuggingFace
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        max_samples_per_class: Maximum samples per class
    """
    print("=" * 60)
    print("BIRD CLASSIFIER TRAINING")
    print("=" * 60)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Download and prepare dataset
    train_images, train_labels, test_images, test_labels, class_names = \
        download_and_prepare_dataset(max_samples_per_class=max_samples_per_class)

    num_classes = len(class_names)
    print(f"\nNumber of classes: {num_classes}")

    # Initialize processor and model
    print(f"\nLoading base model: {model_name}")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )

    # Update model config with class names
    model.config.id2label = {i: name for i, name in enumerate(class_names)}
    model.config.label2id = {name: i for i, name in enumerate(class_names)}

    # Create datasets
    train_dataset = BirdDataset(train_images, train_labels, processor, class_names)
    test_dataset = BirdDataset(test_images, test_labels, processor, class_names)

    print(f"\nTrain dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, 'checkpoints'),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="none",
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    train_result = trainer.train()

    # Evaluate the model
    print("\n" + "=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)
    eval_result = trainer.evaluate()

    # Get predictions for confusion matrix
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = test_labels

    # Calculate additional metrics
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Per-class accuracy
    per_class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    per_class_metrics = {
        class_names[i]: {
            'accuracy': float(per_class_accuracy[i]) if not np.isnan(per_class_accuracy[i]) else 0.0,
            'samples': int(conf_matrix[i].sum())
        }
        for i in range(len(class_names))
    }

    # Prepare metrics dictionary
    metrics = {
        'model_name': model_name,
        'training_date': datetime.now().isoformat(),
        'num_classes': num_classes,
        'num_train_samples': len(train_dataset),
        'num_test_samples': len(test_dataset),
        'hyperparameters': {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'optimizer': 'AdamW',
            'warmup_steps': 100,
            'weight_decay': 0.01,
        },
        'results': {
            'test_accuracy': float(eval_result['eval_accuracy']),
            'test_precision': float(eval_result['eval_precision']),
            'test_recall': float(eval_result['eval_recall']),
            'test_f1': float(eval_result['eval_f1']),
            'train_loss': float(train_result.training_loss),
        },
        'per_class_metrics': per_class_metrics,
        'class_names': class_names,
    }

    # Save metrics
    metrics_path = os.path.join(output_dir, 'model_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✓ Metrics saved to {metrics_path}")

    # Save the model
    model_path = os.path.join(output_dir, 'bird_classifier')
    trainer.save_model(model_path)
    processor.save_pretrained(model_path)

    print(f"✓ Model saved to {model_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Test Accuracy: {metrics['results']['test_accuracy']:.4f}")
    print(f"Test Precision: {metrics['results']['test_precision']:.4f}")
    print(f"Test Recall: {metrics['results']['test_recall']:.4f}")
    print(f"Test F1 Score: {metrics['results']['test_f1']:.4f}")
    print("\nTop 5 Best Performing Classes:")
    sorted_classes = sorted(
        per_class_metrics.items(),
        key=lambda x: x[1]['accuracy'],
        reverse=True
    )[:5]
    for class_name, metrics_dict in sorted_classes:
        print(f"  {class_name}: {metrics_dict['accuracy']:.4f} ({metrics_dict['samples']} samples)")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train bird classifier model')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max-samples', type=int, default=100, help='Max samples per class')
    parser.add_argument('--output-dir', type=str, default='backend/models', help='Output directory')

    args = parser.parse_args()

    # Check for CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("⚠️  Warning: Training on CPU will be slow. Consider using GPU for faster training.")

    # Train the model
    train_model(
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_samples_per_class=args.max_samples
    )
