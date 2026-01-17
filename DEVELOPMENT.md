# Development Guide

## Backend Development

### Setup
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running
```bash
python main.py
```

The server will start at `http://localhost:8000`

### API Endpoints

#### Health & Info
- `GET /` - Welcome message
- `GET /health` - Health check
- `GET /api/species` - List all bird species

#### Classification
- `POST /api/upload` - Upload image (returns base64)
- `POST /api/classify` - Classify bird from image

Request:
```bash
curl -X POST -F "file=@bird.jpg" http://localhost:8000/api/classify
```

Response:
```json
{
  "species": "Hawk",
  "confidence": 0.406,
  "top_predictions": [
    {"species": "Hawk", "confidence": 0.406},
    {"species": "Crow", "confidence": 0.217}
  ]
}
```

#### Augmentation
- `POST /api/augment` - Add accessories to bird image

Request:
```json
{
  "image_base64": "base64_encoded_image",
  "edit_type": "hat"
}
```

Supported edit types: `hat`, `bowtie`, `glasses`, `crown`, `scarf`

### ML Models

The backend uses:
- **Vision Transformer** from HuggingFace for classification
- **Stable Diffusion** for image inpainting (demo mode available)

Models are lazily loaded on first use. Demo mode works without downloading large models.

## Frontend Development

### Setup
```bash
cd frontend
npm install
```

### Running
```bash
npm run dev
```

The app will start at `http://localhost:3000`

### Building
```bash
npm run build
```

Output in `dist/` directory.

### Project Structure

```
frontend/src/
├── components/
│   ├── ImageUpload.tsx      # Image upload with drag & drop
│   ├── BirdCard.tsx          # Bird display with augmentation
│   └── CollectionProgress.tsx # Progress tracking
├── App.tsx                   # Main application
├── api.ts                    # API client
└── types.ts                  # TypeScript types
```

## Testing

### Backend Tests
```bash
cd backend
source venv/bin/activate

# Test health endpoint
curl http://localhost:8000/health

# Test classification
curl -X POST -F "file=@test_bird.jpg" http://localhost:8000/api/classify

# Test species list
curl http://localhost:8000/api/species
```

### Frontend Tests
Open `http://localhost:3000` in your browser and:
1. Upload a bird image
2. Verify classification appears
3. Click accessory buttons to test augmentation
4. Check collection progress updates

## Production Deployment

### Backend
```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Frontend
```bash
npm run build
# Serve the dist/ folder with your preferred web server
```

## Environment Variables

### Backend
- `CUDA_VISIBLE_DEVICES` - GPU device ID (if using CUDA)
- `HF_HOME` - HuggingFace cache directory (optional)

### Frontend
- Create `.env` file for custom API URL:
```
VITE_API_URL=http://your-backend-url:8000
```

## Troubleshooting

### Backend Issues

**Issue**: Models not loading
- Check internet connection for HuggingFace downloads
- Ensure sufficient disk space for models
- Demo mode works without models

**Issue**: CUDA out of memory
- Use CPU mode (automatic fallback)
- Reduce batch size or image resolution

### Frontend Issues

**Issue**: API connection failed
- Verify backend is running on port 8000
- Check CORS settings in backend
- Confirm proxy settings in vite.config.ts

**Issue**: Image upload fails
- Check file size (keep under 10MB)
- Verify file format (JPEG, PNG supported)

## Performance Tips

1. **Backend**: Use GPU for faster inference
2. **Frontend**: Optimize images before upload
3. **Production**: Use CDN for frontend assets
4. **Caching**: Models are cached after first load

## Contributing

1. Follow existing code style
2. Add TypeScript types for new features
3. Test API endpoints before committing
4. Update documentation for new features
