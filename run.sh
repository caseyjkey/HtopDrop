#!/bin/bash
# HtopDrop launcher script

set -e

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

# Check if dependencies are installed
echo "Checking dependencies..."
python3 -c "import pygame" 2>/dev/null || {
    echo "Installing dependencies..."
    pip install -r requirements.txt --user
}

# Run HtopDrop
echo "Starting HtopDrop..."
python3 htop_drop.py "$@"
