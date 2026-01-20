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
echo "📦 Step 1: Installing dependencies to virtual environment..."
cd backend

# Check if venv exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "✓ Virtual environment already exists"
fi

source venv/bin/activate

# Check if requirements are installed
if python3 -c "import torch, transformers, fastapi" 2>/dev/null; then
    echo "✓ Dependencies already installed"
else
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Step 2: Check for existing models
echo ""
echo "🎓 Step 2: Checking for existing trained models..."
echo ""

# Check if models directory exists and has trained models
if [ -d "models" ]; then
    # Find all trained models (both timestamped and non-timestamped)
    models_found=()
    
    # Check for timestamped models (bird_classifier_YYYY-MM-DD_HH-MM-SS)
    for model_dir in models/bird_classifier_*; do
        if [ -d "$model_dir" ]; then
            # Check if it has config.json (direct structure) or bird_classifier/config.json (nested)
            if [ -f "$model_dir/config.json" ] || [ -f "$model_dir/bird_classifier/config.json" ]; then
                models_found+=("$model_dir")
            fi
        fi
    done
    
    # Check for non-timestamped model
    if [ -d "models/bird_classifier" ] && [ -f "models/bird_classifier/config.json" ]; then
        models_found+=("models/bird_classifier")
    fi
    
    # Display found models
    if [ ${#models_found[@]} -gt 0 ]; then
        echo "Found ${#models_found[@]} trained model(s):"
        echo ""
        
        model_index=1
        for model_path in "${models_found[@]}"; do
            echo "[$model_index] $model_path"
            
            # Try to extract metrics if available
            metrics_file=""
            if [ -f "$model_path/model_metrics.json" ]; then
                metrics_file="$model_path/model_metrics.json"
            elif [ -f "$(dirname "$model_path")/model_metrics.json" ]; then
                metrics_file="$(dirname "$model_path")/model_metrics.json"
            fi
            
            if [ -n "$metrics_file" ]; then
                # Extract timestamp and accuracy using python
                python3 -c "
import json
try:
    with open('$metrics_file', 'r') as f:
        data = json.load(f)
    timestamp = data.get('training_date', 'N/A')
    accuracy = data.get('results', {}).get('test_accuracy', 0)
    print(f'    Timestamp: {timestamp}')
    print(f'    Accuracy: {accuracy*100:.1f}%')
except:
    pass
" 2>/dev/null
            fi
            echo ""
            model_index=$((model_index + 1))
        done
        
        # Prompt user to select a model
        echo "Would you like to use one of these models?"
        read -p "Enter model number (1-${#models_found[@]}) or press Enter to train a new model: " model_choice
        
        if [ -n "$model_choice" ] && [ "$model_choice" -ge 1 ] && [ "$model_choice" -le ${#models_found[@]} ]; then
            echo "✓ Using existing model: ${models_found[$((model_choice-1))]}"
            skip_training=true
        else
            echo "Proceeding to train a new model..."
            skip_training=false
        fi
    else
        echo "No trained models found."
        skip_training=false
    fi
else
    echo "No models directory found."
    skip_training=false
fi

# Step 3: Train the model (if not using existing)
if [ "$skip_training" != "true" ]; then
    echo ""
    echo "🎓 Training the model..."
    echo ""
    read -p "Choose training mode (1=Quick test, 2=Standard, 3=High quality, 4=Skip training - use base model): " mode

    case $mode in
        1)
            echo "Running quick test (1 epochs, 20 samples/class, batch size 4)..."
            python3 train_model.py --epochs 1 --max-samples 20 --batch-size 4
            ;;
        2)
            echo "Running standard training (5 epochs, 100 samples/class, batch size 16)..."
            python3 train_model.py --epochs 5 --max-samples 100 --batch-size 16
            ;;
        3)
            echo "Running high quality training (10 epochs, 200 samples/class, batch size 32)..."
            python3 train_model.py --epochs 10 --max-samples 200 --batch-size 32
            ;;
        4)
            echo "Skipping training - app will use base model (no fine-tuning)"
            ;;
        *)
            echo "Invalid choice - skipping training, app will use base model"
            ;;
    esac
fi

# Step 3: Start the backend
echo ""
echo "🚀 Step 3: Starting the backend..."
echo "Backend will run at http://localhost:8000"
echo ""
python3 main.py &
BACKEND_PID=$!

# Wait a bit for backend to start
sleep 10

# Step 4: Start the frontend
echo ""
echo "🎨 Step 4: Starting the frontend..."
echo "Frontend will run at http://localhost:5173"
echo ""
cd ../frontend

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
echo "Frontend: http://localhost:3000"
echo "Backend: http://localhost:8000"
echo "Model Stats: Click '📊 Model Stats' in the navigation"
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for user to press Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait
