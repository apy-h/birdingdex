# Fine-Tuning Guide for Birdingdex

This guide explains how to fine-tune the bird classification model on the OpenML bird dataset.

## Overview

The Birdingdex application now supports fine-tuning a Vision Transformer (ViT) model on the OpenML Birds 525 Species dataset (ID: 44320). This fine-tuned model will replace the general-purpose vision model and provide more accurate bird species classification.

## Prerequisites

1. **Python Environment**: Ensure you have Python 3.8+ installed
2. **Virtual Environment**: Activate your backend virtual environment
3. **Dependencies**: Install all required packages

```bash
cd backend
pip install -r requirements.txt
```

## Quick Start

### Training the Model

Run the training script with default parameters:

```bash
cd backend
python train_model.py
```

This will:
- Download the OpenML bird dataset (ID: 44320)
- Fine-tune a Vision Transformer model
- Train for 5 epochs with a batch size of 16
- Save the model to `backend/models/bird_classifier/`
- Save training metrics to `backend/models/model_metrics.json`

### Custom Training Parameters

You can customize the training process with command-line arguments:

```bash
python train_model.py \
  --epochs 10 \
  --batch-size 32 \
  --learning-rate 1e-5 \
  --max-samples 200 \
  --output-dir backend/models
```

**Available Options:**
- `--epochs`: Number of training epochs (default: 5)
- `--batch-size`: Batch size for training (default: 16)
- `--learning-rate`: Learning rate (default: 2e-5)
- `--max-samples`: Max samples per class for faster training (default: 100)
- `--output-dir`: Directory to save the model (default: backend/models)

## Dataset Information

### OpenML Birds Dataset (ID: 44320)

- **Dataset**: Birds 525 Species - Image Classification
- **Source**: OpenML (https://www.openml.org/d/44320)
- **Content**: Bird images with species labels
- **Classes**: Multiple bird species from around the world

### Dataset Access

The training script automatically downloads the dataset using the `scikit-learn` OpenML API:

```python
from sklearn.datasets import fetch_openml
data = fetch_openml(data_id=44320, as_frame=False, parser='auto')
```

**Note**: The first download may take several minutes depending on your internet connection. The dataset will be cached locally for subsequent runs.

### Fallback: Synthetic Dataset

If the OpenML download fails, the script automatically falls back to a synthetic dataset with common bird species. This is useful for testing the training pipeline without downloading large datasets.

## Training Process

### 1. Data Preparation

- Downloads bird images from OpenML
- Balances the dataset (limits samples per class)
- Splits into 80% training, 20% testing
- Applies image preprocessing for Vision Transformer

### 2. Model Architecture

- **Base Model**: `google/vit-base-patch16-224`
- **Type**: Vision Transformer (ViT)
- **Input Size**: 224x224 pixels
- **Fine-tuning**: Transfer learning from pre-trained ImageNet weights

### 3. Training Configuration

Default hyperparameters:
- **Optimizer**: AdamW
- **Learning Rate**: 2e-5
- **Warmup Steps**: 100
- **Weight Decay**: 0.01
- **Batch Size**: 16
- **Epochs**: 5

### 4. Evaluation Metrics

The model is evaluated on:
- **Accuracy**: Overall classification accuracy
- **Precision**: Weighted average precision
- **Recall**: Weighted average recall
- **F1 Score**: Weighted average F1 score
- **Per-Class Accuracy**: Individual performance for each species

## Output Files

After training, you'll find:

### 1. Model Directory: `backend/models/bird_classifier/`
Contains:
- `config.json`: Model configuration
- `pytorch_model.bin`: Trained model weights
- `preprocessor_config.json`: Image preprocessing configuration
- `model.safetensors`: Safe tensors format (if available)

### 2. Metrics File: `backend/models/model_metrics.json`
Contains:
```json
{
  "model_name": "google/vit-base-patch16-224",
  "training_date": "2026-01-16T...",
  "num_classes": 20,
  "num_train_samples": 800,
  "num_test_samples": 200,
  "hyperparameters": {
    "num_epochs": 5,
    "batch_size": 16,
    "learning_rate": 2e-5,
    ...
  },
  "results": {
    "test_accuracy": 0.85,
    "test_precision": 0.84,
    "test_recall": 0.83,
    "test_f1": 0.84
  },
  "per_class_metrics": {
    "American Robin": {"accuracy": 0.92, "samples": 40},
    ...
  }
}
```

## Using the Fine-Tuned Model

### Backend Integration

The fine-tuned model is automatically loaded when you start the backend:

```bash
cd backend
python main.py
```

The `BirdClassifier` in `ml_service.py` will:
1. Check for fine-tuned model at `backend/models/bird_classifier/`
2. Load the fine-tuned model if available
3. Fall back to the base model if not found

### Frontend Model Stats Page

View training metrics and model performance:
1. Start the backend and frontend
2. Navigate to the "Model Stats" page
3. View:
   - Overall performance metrics (accuracy, precision, recall, F1)
   - Hyperparameters used during training
   - Top and bottom performing bird species
   - Training date and dataset information

## Performance Tips

### 1. GPU Acceleration

For faster training, use a GPU:
- The script automatically detects and uses CUDA if available
- Training on CPU is slower but still functional
- Expected training time:
  - **GPU**: 10-20 minutes (5 epochs, 100 samples/class)
  - **CPU**: 1-2 hours (5 epochs, 100 samples/class)

### 2. Memory Management

If you encounter out-of-memory errors:
- Reduce `--batch-size` (try 8 or 4)
- Reduce `--max-samples` (try 50)
- Close other applications

### 3. Training Time vs. Accuracy

Balance between training time and model quality:
- **Quick test**: `--epochs 3 --max-samples 50` (~5 min on GPU)
- **Balanced**: `--epochs 5 --max-samples 100` (~15 min on GPU)
- **High quality**: `--epochs 10 --max-samples 200` (~45 min on GPU)

## Troubleshooting

### OpenML Download Fails

**Error**: `Failed to download dataset from OpenML`

**Solution**: The script automatically uses a synthetic dataset. If you need the real dataset:
1. Check your internet connection
2. Try again later (OpenML may be temporarily down)
3. Manually download from: https://www.openml.org/d/44320

### Out of Memory

**Error**: `CUDA out of memory` or `RuntimeError: out of memory`

**Solution**:
```bash
python train_model.py --batch-size 4 --max-samples 50
```

### Model Not Loading in Backend

**Error**: Backend shows "Demo mode" or "Model not found"

**Solution**:
1. Check that training completed successfully
2. Verify model exists at `backend/models/bird_classifier/`
3. Check file permissions
4. Restart the backend server

## Advanced Usage

### Re-training the Model

To retrain with new parameters:
```bash
# Remove old model
rm -rf backend/models/bird_classifier
rm backend/models/model_metrics.json

# Train with new parameters
python train_model.py --epochs 10 --batch-size 32
```

### Custom Model Path

Save the model to a custom location:
```bash
python train_model.py --output-dir /path/to/custom/location
```

Then update `ml_service.py`:
```python
bird_classifier = BirdClassifier(model_path="/path/to/custom/location/bird_classifier")
```

### Training Monitoring

Monitor training progress:
- Console output shows epoch progress
- Loss values are printed per epoch
- Evaluation metrics shown after each epoch

## Next Steps

After training:
1. Start the backend: `python main.py`
2. Upload bird images to test classification
3. View model stats to analyze performance
4. If accuracy is low, retrain with more epochs or samples

## Additional Resources

- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [OpenML Documentation](https://www.openml.org/guide)
- [Transfer Learning Guide](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
