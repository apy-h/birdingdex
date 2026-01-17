# 🦅 Birdingdex

A Pokédex for Birds! A full-stack web application that uses AI to identify bird species from photos and lets you build your own bird collection.

## ✨ Features

- 🤖 **AI-Powered Classification**: Upload bird photos and get instant species identification using Vision Transformer models
- � **Fine-Tuned Model**: Train your own bird classifier on the OpenML Birds dataset (525 species)
- 📊 **Model Statistics**: View detailed performance metrics, accuracy, and hyperparameters
- 🎨 **Image Augmentation**: Add fun accessories to your bird photos (hats, bowties, glasses) using Stable Diffusion inpainting
- 📈 **Collection Tracking**: Build your personal bird collection and track your progress
- 🎯 **Modern UI**: Beautiful, responsive React interface with TypeScript
- ⚡ **Fast API**: High-performance FastAPI backend with async support

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
- **Vision Transformer (ViT)** fine-tuned on OpenML bird dataset
- **Stable Diffusion** via diffusers for image inpainting
- **PyTorch** for deep learning
- **scikit-learn** for dataset loading and metrics
- OpenML Birds 525 Species dataset (ID: 44320)

## 📋 Prerequisites

- Python 3.8+ (for backend)
- Node.js 18+ (for frontend)
- pip (Python package manager)
- npm or yarn (Node package manager)

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

## � Fine-Tuning the Model (Optional)

For best results, fine-tune the model on the OpenML bird dataset:

```bash
cd backend
python train_model.py
```

**Quick training** (for testing):
```bash
python train_model.py --epochs 3 --max-samples 50
```

**Production quality**:
```bash
python train_model.py --epochs 10 --max-samples 200
```

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed instructions.

## 🎮 Usage

1. **Upload a Bird Photo**: Click or drag-and-drop a bird image onto the upload area
2. **Get Identification**: The AI will classify the bird species with confidence scores
3. **View Model Stats**: Click "📊 Model Stats" to see model performance and metrics
4. **View Your Collection**: See all the birds you've discovered in your collection grid
5. **Track Progress**: Monitor how many species you've found
6. **Add Accessories** (optional): Click the accessory buttons to add fun cosmetic edits to your bird photos

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
│   │   │   └── CollectionProgress.tsx
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

### Backend Development
```bash
cd backend
python main.py
```

The API will auto-reload on code changes with uvicorn's auto-reload feature.

### Frontend Development
```bash
cd frontend
npm run dev
```

Vite provides hot module replacement for instant updates.

### Building for Production

Frontend:
```bash
cd frontend
npm run build
```

The production build will be in `frontend/dist/`.

## 🎯 Key Features Explained

### Bird Classification
- Uses pre-trained Vision Transformer models from HuggingFace
- Provides top-5 predictions with confidence scores
- Supports 25+ common bird species
- Extensible to OpenML bird dataset for more species

### Image Augmentation
- Stable Diffusion inpainting for realistic edits
- Pre-configured prompts for common accessories
- Supports custom prompts
- Demo mode with text overlays when SD models aren't loaded

### Collection System
- Tracks unique bird species discovered
- Shows progress as a percentage
- Displays collection grid with all discoveries
- Celebration message on 100% completion

## 🔧 Configuration

### Backend Configuration
- Server runs on port 8000 by default
- CORS enabled for localhost:3000 and localhost:5173
- Models use CUDA if available, otherwise CPU

### Frontend Configuration
- Development server on port 3000
- API proxy configured in vite.config.ts
- TypeScript strict mode enabled

## 📝 Notes

- **Demo Mode**: The application includes a demo mode that works without downloading large ML models
- **Model Loading**: First run may take time to download HuggingFace models
- **GPU Acceleration**: CUDA-enabled GPU recommended for faster inference
- **Image Formats**: Supports common image formats (JPEG, PNG, WebP, etc.)

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is open source and available for educational and demonstration purposes.

## 🙏 Acknowledgments

- HuggingFace for transformer models and diffusers
- FastAPI for the excellent web framework
- React and Vite for modern frontend development
- OpenML for bird datasets
