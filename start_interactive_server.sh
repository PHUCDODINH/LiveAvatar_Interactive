#!/bin/bash
# Interactive LiveAvatar Server - Startup Script

echo "=========================================="
echo "Starting Interactive LiveAvatar Server"
echo "=========================================="

# Activate conda environment if needed
if [ ! -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Using conda environment: $CONDA_DEFAULT_ENV"
fi

# Set environment variables for optimization
export ENABLE_COMPILE=true
export ENABLE_FP8=true
export CUDNN_BENCHMARK=1
export CUDA_VISIBLE_DEVICES=0

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found!"
    echo "Please copy .env.example to .env and fill in your API keys"
    exit 1
fi

# Check if dependencies are installed
echo "Checking dependencies..."
python -c "import fastapi, uvicorn, openai, elevenlabs" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing additional dependencies..."
    pip install -r requirements_interactive.txt
fi

# Create output directory
mkdir -p output/interactive

echo ""
echo "Starting server..."
echo "Server will be available at: http://0.0.0.0:8000"
echo "Press Ctrl+C to stop"
echo ""

# Start the server
python interactive_avatar_server.py
