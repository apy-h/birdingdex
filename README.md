# 🦅 Birdingdex

A Pokédex for Birds! A full-stack web application that uses AI to identify bird species from photos and lets you build your own bird collection.

## ✨ Features

- 🤖 **AI-Powered Classification**: Upload bird photos and get instant species identification using Vision Transformer models
- 🎯 **Fine-Tuned Model**: Train your own bird classifier on the CUB-200-2011 dataset, which supports 200 bird species
- 📊 **Model Statistics**: View detailed performance metrics, accuracy, and hyperparameters
- 🎨 **Image Augmentation**: Add fun accessories to your bird photos (hats, bowties, glasses) using Stable Diffusion inpainting
- 📈 **Collection Tracking**: Build your personal bird collection and track your progress
- 🎯 **Modern UI**: Beautiful, responsive React interface with TypeScript
- ⚡ **Fast API**: High-performance FastAPI backend with async support

## 🚀 Quick Start

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the FastAPI server:
```bash
python main.py
```

The backend will be running at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The frontend will be running at `http://localhost:3000`

### Training the Model

By default, the app uses the pre-trained base Vision Transformer (`google/vit-base-patch16-224` from HuggingFace). To improve accuracy for all 200 bird species, you can optionally fine-tune it on the CUB-200-2011 bird dataset:

**Quick test** (CPU-friendly, ~10–15 min):
```bash
cd backend
python train_model.py --epochs 1 --max-samples 20 --batch-size 4
```

**Standard training** (full dataset):
```bash
python train_model.py --epochs 10 --max-samples 200 --batch-size 8
```

The script automatically downloads the dataset via Kaggle API or falls back to direct download. See the [Development](#-development) section for Kaggle API setup and troubleshooting.

## 🎮 Usage

1. **Upload a Bird Photo**: Click or drag-and-drop a bird image onto the upload area
2. **Get Identification**: The AI will classify the bird species with confidence scores
3. **View Model Stats**: Click "📊 Model Stats" to see model performance and metrics
4. **View Your Collection**: See all the birds you've discovered in your collection grid
5. **Track Progress**: Monitor how many species you've found
6. **Add Accessories** (optional): Click the accessory buttons to add fun cosmetic edits to your bird photos

## 🏗️ Tech Stack

### Frontend
- **React** with TypeScript
- **Vite** for blazing-fast development
- **Axios** for API communication
- Modern CSS with custom properties

### Backend
- **FastAPI** (Python) for REST APIs
- **Uvicorn** ASGI server
- **CORS** middleware for frontend integration

### ML/AI
- **HuggingFace Transformers** for bird classification
- **Vision Transformer (ViT)** fine-tuned on CUB-200-2011 (Caltech-UCSD Birds-200-2011) bird images
- **Stable Diffusion** via diffusers for image inpainting
- **PyTorch** for deep learning
- **Kaggle API** and direct download for dataset acquisition

## 🔌 API Endpoints

### Health & Info
- `GET /` - API welcome message
- `GET /health` - Health check
- `GET /api/species` - Get list of all supported bird species
- `GET /api/model/metrics` - Get model training metrics and performance stats

### Bird Classification
- `POST /api/upload` - Upload and validate an image
- `POST /api/classify` - Classify bird species from image

### Image Augmentation
- `POST /api/augment` - Apply cosmetic edits to bird images

## 📁 Project Structure

```
birdingdex/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── ml_service.py        # ML models and inference
│   ├── train_model.py       # Model fine-tuning script
│   ├── requirements.txt     # Python dependencies
│   └── models/              # Saved models directory
│       ├── bird_classifier/ # Fine-tuned model
│       └── model_metrics.json
├── frontend/
│   ├── src/
│   │   ├── components/      # React components
│   │   │   ├── ImageUpload.tsx
│   │   │   ├── BirdCard.tsx
│   │   │   ├── CollectionProgress.tsx
│   │   │   └── ModelStats.tsx
│   │   ├── App.tsx          # Main app component
│   │   ├── api.ts           # API client
│   │   ├── types.ts         # TypeScript types
│   │   └── main.tsx         # Entry point
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts
│   └── tsconfig.json
└── README.md
```

## 🧪 Development

### Backend

#### Custom Training Parameters

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

#### Training Process

##### 1. Data Preparation

- Uses CUB-200-2011 bird images
- `train_model.py` downloads via (1) Kaggle API if `KAGGLE_API_TOKEN` or `~/.kaggle/kaggle.json` is set; (2) direct download fallback (no credentials)
- Local cache: `dataset/` (or `backend/dataset/`), gitignored
- Balances the dataset (limits samples per class)
- Splits into 80% training, 20% testing
- Applies image preprocessing for Vision Transformer

##### 2. Model Architecture

- **Base Model**: `google/vit-base-patch16-224`
- **Type**: Vision Transformer (ViT)
- **Input Size**: 224x224 pixels
- **Fine-tuning**: Transfer learning from pre-trained ImageNet weights

##### 3. Training Configuration

Default hyperparameters:
- **Optimizer**: AdamW
- **Learning Rate**: 2e-5
- **Warmup Steps**: 100
- **Weight Decay**: 0.01
- **Batch Size**: 16
- **Epochs**: 5

**Common training configurations**:
- **Quick test** (CPU-friendly): `python train_model.py --epochs 1 --max-samples 20 --batch-size 4` (~10–15 min)
- **Standard**: `python train_model.py --epochs 5 --max-samples 100 --batch-size 16` (~30–60 min on GPU, ~2 hours on CPU)
- **High quality**: `python train_model.py --epochs 10 --max-samples 200 --batch-size 32` (~2 hours on GPU)

##### 4. Evaluation Metrics

The model is evaluated on:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Per-Class Accuracy**: Individual performance for each species

#### Output Files

After training, you'll find:

##### 1. Model Directory: `backend/models/bird_classifier/`
Contains:
- `config.json`: Model configuration
- `pytorch_model.bin`: Trained model weights
- `preprocessor_config.json`: Image preprocessing configuration
- `model.safetensors`: Safe tensors format (if available)

##### 2. Metrics File: `backend/models/model_metrics.json`
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

**Environment Variables**:
- `KAGGLE_API_TOKEN` (preferred) or `~/.kaggle/kaggle.json` for dataset download
- `HF_HOME` optional HuggingFace cache path
- `CUDA_VISIBLE_DEVICES` if using CUDA-capable GPU


**Testing**:
```bash
curl http://localhost:8000/health
curl -X POST -F "file=@test_bird.jpg" http://localhost:8000/api/classify
curl http://localhost:8000/api/model/metrics
```

**Configuration**:
- Runs on port 8000 by default
- CORS enabled for localhost:3000 and localhost:5173
- Models use CUDA if available, otherwise CPU

**Performance notes**:
- GPU (NVIDIA + CUDA) provides 5–10× speedup over CPU
- Reduce `--batch-size` or `--max-samples` if running out of memory
- Training time grows with dataset size and epochs

**Troubleshooting**:
- **Kaggle**: export `KAGGLE_API_TOKEN=...` or create `~/.kaggle/kaggle.json`, then rerun training
- **Download fails**: script falls back to direct download automatically
- **Memory issues**: reduce `--batch-size` or `--max-samples`
- **Model not loading**: verify training completed and model exists at `backend/models/bird_classifier/`

### Frontend

**Build**:
```bash
npm run build
```
Outputs to `frontend/dist/`.

**Environment**:
- Set `VITE_API_URL` in frontend `.env` if backend URL differs from default

**UI**:
- `ModelStats` component pulls `/api/model/metrics` to display accuracy/loss and per-class stats

**Testing**:
- open `http://localhost:3000`, upload an image, verify classification and model stats.

**Configuration**:
- Development server on port 3000
- API proxy configured in vite.config.ts
- TypeScript strict mode enabled