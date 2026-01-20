#!/bin/bash
# Build script for Render deployment
# Builds React frontend and copies to backend/dist for serving

set -e  # Exit on error

echo "📦 Building frontend..."
cd frontend
npm install
npm run build
cd ..

echo "📋 Copying frontend build to backend..."
rm -rf backend/dist
cp -r frontend/dist backend/dist

echo "⚙️ Installing backend dependencies..."
cd backend
pip install --no-cache-dir --prefer-binary -r requirements.txt

echo "📥 Downloading pre-trained model from HuggingFace..."
# Download model from ap-y/birdingdex-classifier
python3 -c "
import os
from huggingface_hub import snapshot_download

try:
    models_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    print('Downloading model from ap-y/birdingdex-classifier...')
    snapshot_download(
        repo_id='ap-y/birdingdex-classifier',
        local_dir=os.path.join(models_dir, 'bird_classifier'),
        repo_type='model'
    )
    print('✓ Model downloaded successfully to models/bird_classifier')
except Exception as e:
    print(f'Warning: Could not download model from HuggingFace: {e}')
    print('App will fall back to base model')
"
cd ..

echo "✅ Build complete!"
