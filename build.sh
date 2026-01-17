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
pip install -r requirements.txt
cd ..

echo "✅ Build complete!"
