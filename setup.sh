#!/bin/bash
# ATS Backend Setup Script
# This script sets up the development environment for Linux/Ubuntu

echo "========================================"
echo "  ATS Backend - Setup Script (Linux)"
echo "========================================"
echo

# Check Python version
echo "[INFO] Checking Python version..."
if ! python3 --version >/dev/null 2>&1; then
    echo "[ERROR] Python 3 is not installed or not in PATH!"
    echo "Please install Python 3.10 or higher."
    echo
    read -p "Press Enter to exit..." _
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtual environment..."
    if ! python3 -m venv venv; then
        echo "[ERROR] Failed to create virtual environment!"
        read -p "Press Enter to exit..." _
        exit 1
    fi
    echo "[SUCCESS] Virtual environment created."
else
    echo "[INFO] Virtual environment already exists."
fi

# Activate virtual environment
echo "[INFO] Activating virtual environment..."
if [ -f "venv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source venv/bin/activate
else
    echo "[ERROR] Could not find venv/bin/activate"
    read -p "Press Enter to exit..." _
    exit 1
fi

# Upgrade pip
echo "[INFO] Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
echo "[INFO] Installing dependencies..."
if ! pip install -r requirements.txt; then
    echo "[ERROR] Failed to install dependencies!"
    read -p "Press Enter to exit..." _
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "[INFO] Creating .env file from .env.example..."
    if [ -f ".env.example" ]; then
        cp ".env.example" ".env"
        echo "[SUCCESS] .env file created. Please edit it with your configuration."
    else
        echo "[WARNING] .env.example not found. Please create .env manually."
    fi
else
    echo "[INFO] .env file already exists."
fi

echo
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo
echo "Next steps:"
echo "1. Edit .env file with your MySQL credentials and OLLAMA settings"
echo "2. Ensure MySQL server is running"
echo "3. Run database migrations: alembic upgrade head"
echo "4. Ensure OLLAMA is running with required models"
echo "5. Start the API server: ./1start_api.sh"
echo
read -p "Press Enter to exit..." _


