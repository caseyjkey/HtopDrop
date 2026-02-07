#!/bin/bash
# Installation script for Gilbert (Arch Linux)

set -e

echo "Installing HtopDrop on Gilbert (Arch Linux)"
echo "==========================================="
echo

# Check if running on Arch
if [ ! -f /etc/arch-release ]; then
    echo "Warning: This script is designed for Arch Linux"
    echo "You may need to install dependencies manually"
    echo
fi

# Install system dependencies
echo "[1/4] Installing system packages..."
if command -v pacman &> /dev/null; then
    sudo pacman -S --needed python python-pip python-pygame portaudio || {
        echo "Note: Some packages may already be installed"
    }
else
    echo "Warning: pacman not found. Please install manually:"
    echo "  - python"
    echo "  - python-pip"
    echo "  - python-pygame"
    echo "  - portaudio"
fi

echo

# Create virtual environment (optional but recommended)
echo "[2/4] Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created"
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

echo

# Install Python dependencies
echo "[3/4] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo

# Test imports
echo "[4/4] Testing installation..."
python3 -c "
import sys
try:
    import pygame
    import psutil
    import sounddevice
    import numpy
    print('✓ All dependencies installed successfully')
    sys.exit(0)
except ImportError as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo
    echo "==========================================="
    echo "Installation complete!"
    echo
    echo "To run HtopDrop:"
    echo "  source venv/bin/activate  # Activate virtual environment"
    echo "  ./run.sh                   # Run HtopDrop"
    echo
    echo "Or directly:"
    echo "  python3 htop_drop.py       # Run without virtual environment"
    echo
    echo "Options:"
    echo "  --fullscreen              # Run in fullscreen mode"
    echo "  --list-devices            # List audio devices"
    echo "  --audio-device N          # Use specific audio device"
    echo "  --debug                   # Debug mode (console output only)"
    echo "==========================================="
else
    echo
    echo "Installation completed with errors."
    echo "Please check the error messages above."
    exit 1
fi
