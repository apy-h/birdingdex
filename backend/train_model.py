"""Training script for fine-tuning a Vision Transformer model on CUB-200-2011 bird dataset.

This script:
1. Downloads the bird dataset from Kaggle
2. Prepares the data for training
3. Fine-tunes a pre-trained Vision Transformer model
4. Evaluates the model and saves metrics
5. Saves the fine-tuned model for inference
"""

print("Starting training script... loading libraries (this can take a few seconds)...", flush=True)

import os
import json
import time
import shutil
from datetime import datetime
import warnings

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer
)

from prep_data import download_and_prepare_dataset

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
    max_samples_per_class=100,
    image_size=224
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
        image_size: Size to resize images to (default: 224)
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
        download_and_prepare_dataset(
            max_samples_per_class=max_samples_per_class,
            image_size=image_size
        )

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
    checkpoints_root = os.path.join(output_dir, 'checkpoints')
    checkpoint_count = 0

    if os.path.isdir(checkpoints_root):
        for item in os.listdir(checkpoints_root):
            item_path = os.path.join(checkpoints_root, item)
            if os.path.isdir(item_path) and item.startswith('checkpoint-'):
                try:
                    shutil.rmtree(item_path)
                    checkpoint_count += 1
                except Exception as e:
                    print(f"  Warning: Failed to delete {item}: {e}")

        # Remove the empty checkpoints root if all subfolders were deleted
        try:
            if not os.listdir(checkpoints_root):
                shutil.rmtree(checkpoints_root)
        except Exception:
            pass

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
    parser.add_argument('--image-size', type=int, default=224, help='Image size for resizing (default: 224)')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR, help='Output directory')
    parser.add_argument('--no-cuda', action='store_true', help='Force CPU and skip CUDA availability check')

    args = parser.parse_args()

    # Check for CUDA (unless user disables)
    if args.no_cuda:
        device = "cpu"
        print("CUDA check disabled via --no-cuda; using CPU")
    else:
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
        max_samples_per_class=args.max_samples,
        image_size=args.image_size
    )
