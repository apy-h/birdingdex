#!/bin/bash
# Quick start script for training and running Birdingdex with fine-tuned model

echo "🦅 Birdingdex Fine-Tuning Quick Start"
echo "======================================"

# Check if we're in the right directory
if [ ! -f "backend/train_model.py" ]; then
    echo "❌ Error: Please run this script from the birdingdex root directory"
    exit 1
fi

# Step 1: Install dependencies
echo ""
echo "📦 Step 1: Installing dependencies..."
cd backend
pip install -r requirements.txt
cd ..

# Step 2: Train the model
echo ""
echo "🎓 Step 2: Training the model..."
echo "This may take 10-20 minutes on GPU, 1-2 hours on CPU"
echo ""
read -p "Choose training mode (1=Quick test, 2=Standard, 3=High quality, 4=Skip): " mode

case $mode in
    1)
        echo "Running quick test (1 epochs, 20 samples/class, batch size 4)..."
        cd backend
        python train_model.py --epochs 1 --max-samples 20 --batch-size 4
        cd ..
        ;;
    2)
        echo "Running standard training (5 epochs, 100 samples/class, batch size 16)..."
        cd backend
        python train_model.py --epochs 5 --max-samples 100 --batch-size 16
        cd ..
        ;;
    3)
        echo "Running high quality training (10 epochs, 200 samples/class, batch size 32)..."
        cd backend
        python train_model.py --epochs 10 --max-samples 200
        cd ..
        ;;
    4)
        echo "Skipping training - app will run in demo mode"
        ;;
    *)
        echo "Invalid choice - skipping training"
        ;;
esac

# Step 3: Start the backend
echo ""
echo "🚀 Step 3: Starting the backend..."
echo "Backend will run at http://localhost:8000"
echo ""
cd backend
python main.py &
BACKEND_PID=$!
cd ..

# Wait a bit for backend to start
sleep 3

# Step 4: Start the frontend
echo ""
echo "🎨 Step 4: Starting the frontend..."
echo "Frontend will run at http://localhost:5173"
echo ""
cd frontend

# Install npm dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install
fi

npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ Birdingdex is running!"
echo "======================================"
echo "Frontend: http://localhost:5173"
echo "Backend: http://localhost:8000"
echo "Model Stats: Click '📊 Model Stats' in the navigation"
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for user to press Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait
