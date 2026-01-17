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

# TODO
# echo "📥 Downloading pre-trained model from HuggingFace..."
# # Replace YOUR_USERNAME with your actual HuggingFace username
# # Replace birdingdex-classifier with your repo name if different
# python -c "
# from huggingface_hub import snapshot_download
# import os
# models_dir = os.path.join(os.getcwd(), 'models')
# snapshot_download(repo_id='YOUR_USERNAME/birdingdex-classifier', local_dir=models_dir, repo_type='model')
# print('✓ Model downloaded successfully')
# "
cd ..

echo "✅ Build complete!"
