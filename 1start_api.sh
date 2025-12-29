#!/bin/bash
# ATS Backend API Startup Script
# This script starts the FastAPI server

echo "========================================"
echo "  ATS Backend API - Starting Server"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -f "venv/bin/activate" ]; then
    echo "[ERROR] Virtual environment not found!"
    echo "Please create it first by running: python3 -m venv venv"
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "[WARNING] .env file not found!"
    echo "Please copy .env.example to .env and configure it."
    echo ""
    read -p "Press Enter to continue..."
fi

# Activate virtual environment
echo "[INFO] Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
python3 -c "import fastapi" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[ERROR] Dependencies not installed!"
    echo "Please run: pip install -r requirements.txt"
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi

# Set environment variables if .env exists
if [ -f ".env" ]; then
    echo "[INFO] Loading environment variables from .env..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# Start the FastAPI server
echo "[INFO] Starting FastAPI server..."
echo "[INFO] Server will be available at: http://localhost:8000"
echo "[INFO] API Documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo "========================================"
echo ""

python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# If we get here, the server stopped
echo ""
echo "[INFO] Server stopped."
read -p "Press Enter to exit..."

