"""Training script for fine-tuning a Vision Transformer model on CUB-200-2011 bird dataset.

This script:
1. Downloads the bird dataset from Kaggle
2. Prepares the data for training
3. Fine-tunes a pre-trained Vision Transformer model
4. Evaluates the model and saves metrics
5. Saves the fine-tuned model for inference
"""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import time
import pickle
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer
)
from datetime import datetime
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


def organize_extracted_dataset(extracted_parent_path, dataset_path):
    """
    Organize extracted dataset to standard location.

    Finds the extracted CUB_200_2011 directory and moves it to the standard location,
    then cleans up the temporary extraction directory.

    Args:
        extracted_parent_path: Path where files were extracted
        dataset_path: Target path for the dataset
    """
    import shutil

    # Find the extracted directory
    extracted_dirs = [d for d in os.listdir(extracted_parent_path)
                      if os.path.isdir(os.path.join(extracted_parent_path, d))]

    if extracted_dirs:
        # Move the extracted dataset to standard location
        src = os.path.join(extracted_parent_path, extracted_dirs[0])
        shutil.move(src, dataset_path)
        print(f"  ✓ Dataset moved to: {dataset_path}")
    else:
        # Files might be directly in the extraction path
        if os.path.exists(extracted_parent_path):
            os.rename(extracted_parent_path, dataset_path)
            print(f"  ✓ Dataset ready at: {dataset_path}")

    # Clean up temporary directory if it still exists
    if os.path.exists(extracted_parent_path):
        shutil.rmtree(extracted_parent_path)


def download_and_prepare_dataset(data_dir=None, max_samples_per_class=100):
    """
    Download and prepare the CUB-200-2011 bird dataset.
    Tries Kaggle API first, then falls back to direct download.

    Args:
        data_dir: Directory to save/cache images (defaults to backend/dataset)
        max_samples_per_class: Maximum samples per class (for faster training)

    Returns:
        Tuple of (train_images, train_labels, test_images, test_labels, class_names)
    """
    # Default to backend/dataset if not specified
    if data_dir is None:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(SCRIPT_DIR, 'dataset')

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

            start_time = time.time()
            kaggle.api.dataset_download_files(
                kaggle_dataset,
                path=kaggle_path,
                unzip=True,
                quiet=False
            )
            elapsed = time.time() - start_time
            print(f"  ✓ Successfully downloaded from Kaggle! ({elapsed:.1f}s)")

            # Move extracted dataset to standard location
            organize_extracted_dataset(kaggle_path, dataset_path)

        except Exception as e:
            print(f"  ✗ Kaggle download failed: {e}")
            print(f"\n  Setup Kaggle API:")
            print(f"    1. Go to https://www.kaggle.com/settings/account")
            print(f"    2. Scroll to 'API' section and click 'Create New Token'")
            print(f"    3. Add this token to your .env file:")
            print(f"       KAGGLE_API_TOKEN=<your-api-token-from-json>")

            # Try Method 2: Direct Download
            download_cub_direct(data_dir, dataset_path)

    # Load the dataset from the standard location
    return load_cub_dataset(dataset_path, data_dir, max_samples_per_class)


def download_cub_direct(data_dir, dataset_path):
    """Direct download from Caltech servers."""
    print("\nMethod 2: Direct download from Caltech...")
    import tarfile
    import urllib.request

    url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
    tgz_path = os.path.join(data_dir, 'CUB_200_2011.tgz')

    print(f"  Downloading from: {url}")
    print(f"  Size: ~1.1 GB - This may take several minutes...")

    try:
        start_time = time.time()
        urllib.request.urlretrieve(url, tgz_path)
        elapsed = time.time() - start_time
        print(f"  ✓ Download complete! ({elapsed:.1f}s)")

        print("  Extracting archive...")
        extract_start = time.time()
        with tarfile.open(tgz_path, 'r:gz') as tar:
            tar.extractall(data_dir)
        extract_elapsed = time.time() - extract_start
        print(f"  ✓ Extraction complete! ({extract_elapsed:.1f}s)")

        # Move extracted dataset to standard location
        organize_extracted_dataset(data_dir, dataset_path)

        # Clean up tar file
        if os.path.exists(tgz_path):
            os.remove(tgz_path)

    except Exception as e:
        print(f"  ✗ Direct download failed: {e}")
        raise Exception("All download methods failed. Please download CUB-200-2011 manually.")


def load_cub_dataset(dataset_path, data_dir, max_samples_per_class):
    """Load CUB-200-2011 dataset from disk with intelligent caching.

    Implements dual-tier caching:
    - Cache 1: All resized images (parameter-independent, ~11,788 images)
    - Cache 2: Balanced subset per max_samples_per_class (parameter-dependent)
    """
    print("\n" + "=" * 60)
    print("LOADING CUB-200-2011 DATASET")
    print("=" * 60)

    load_start = time.time()

    # Define cache paths
    all_images_cache = os.path.join(data_dir, 'cub_cache_resized.pkl')
    balanced_cache = os.path.join(data_dir, f'cub_cache_balanced_max{max_samples_per_class}.pkl')

    # Check Cache 2 first (balanced dataset for this specific max_samples_per_class)
    if os.path.exists(balanced_cache):
        print(f"  Loading balanced dataset from cache...")
        with open(balanced_cache, 'rb') as f:
            cache_data = pickle.load(f)
            X_train = cache_data['X_train']
            X_test = cache_data['X_test']
            y_train = cache_data['y_train']
            y_test = cache_data['y_test']
            class_names = cache_data['class_names']
        load_elapsed = time.time() - load_start
        print(f"  ✓ Loaded balanced dataset from cache! ({load_elapsed:.1f}s)")
        return X_train, y_train, X_test, y_test, class_names

    # Check Cache 1 (all resized images)
    if os.path.exists(all_images_cache):
        print(f"  Loading all resized images from cache...")
        with open(all_images_cache, 'rb') as f:
            cache_data = pickle.load(f)
            all_images = cache_data['images']
            all_labels = cache_data['labels']
            class_names = cache_data['class_names']
        print(f"  ✓ Loaded {len(all_images)} resized images from cache!")
    else:
        # Load from disk
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

        # Save Cache 1 (all resized images)
        print(f"  Saving resized images to cache...")
        with open(all_images_cache, 'wb') as f:
            pickle.dump({
                'images': all_images,
                'labels': all_labels,
                'class_names': class_names
            }, f)
        print(f"  ✓ Cache saved to: {all_images_cache}")

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

    # Save Cache 2 (balanced dataset for this specific max_samples_per_class)
    print(f"\n  Saving balanced dataset to cache...")
    with open(balanced_cache, 'wb') as f:
        pickle.dump({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'class_names': class_names
        }, f)
    print(f"  ✓ Balanced cache saved to: {balanced_cache}")

    load_elapsed = time.time() - load_start
    print(f"\n  ✓ Dataset loading complete! ({load_elapsed:.1f}s)")

    return X_train, y_train, X_test, y_test, class_names


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
    output_dir=None,
    model_name='google/vit-base-patch16-224',
    num_epochs=5,
    batch_size=16,
    learning_rate=2e-5,
    max_samples_per_class=100
):
    """
    Fine-tune a Vision Transformer model on the bird dataset.

    Args:
        output_dir: Directory to save the model and metrics (defaults to backend/models)
        model_name: Pre-trained model name from HuggingFace
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        max_samples_per_class: Maximum samples per class
    """
    # Default to backend/models if not specified
    if output_dir is None:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(SCRIPT_DIR, 'models')

    print("=" * 60)
    print("BIRD CLASSIFIER TRAINING")
    print("=" * 60)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    overall_start = time.time()

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

    # Create timestamped model directory
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    timestamped_model_dir = os.path.join(output_dir, f'bird_classifier_{timestamp}')
    os.makedirs(timestamped_model_dir, exist_ok=True)

    # Save metrics in timestamped directory
    metrics_path = os.path.join(timestamped_model_dir, 'model_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✓ Metrics saved to {metrics_path}")

    # Save the model in timestamped directory
    model_path = os.path.join(timestamped_model_dir, 'bird_classifier')
    trainer.save_model(model_path)
    processor.save_pretrained(model_path)

    print(f"✓ Model saved to {model_path}")

    # Delete checkpoint directories if they exist (safe to delete after training complete)
    # Note: Checkpoint folders contain intermediate epoch saves and are not needed after final model is saved
    print(f"\nCleaning up checkpoint directories...")
    checkpoint_count = 0
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item.startswith('checkpoint-'):
            try:
                shutil.rmtree(item_path)
                checkpoint_count += 1
            except Exception as e:
                print(f"  Warning: Failed to delete {item}: {e}")

    if checkpoint_count > 0:
        print(f"✓ Deleted {checkpoint_count} checkpoint director{'y' if checkpoint_count == 1 else 'ies'}")
    else:
        print(f"  No checkpoint directories found")

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\n✓ Model saved with timestamp: {timestamp}")
    print(f"  Location: {timestamped_model_dir}")
    overall_elapsed = time.time() - overall_start
    print(f"\nTotal training time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} minutes)")
    print()
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

    # Get the directory where this script is located (backend/)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'models')

    parser = argparse.ArgumentParser(description='Train bird classifier model')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max-samples', type=int, default=100, help='Max samples per class')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR, help='Output directory')

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
