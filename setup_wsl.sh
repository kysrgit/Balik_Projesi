#!/bin/bash
# Pufferfish Detection System WSL Setup Script

echo "=========================================="
echo " Pufferfish Detection System - WSL Setup"
echo "=========================================="

REQ_FILE="requirements-dev.txt"
if [ "$1" = "--runtime-only" ]; then
    REQ_FILE="requirements.txt"
elif [ "$1" = "--training" ]; then
    REQ_FILE="requirements-train.txt"
fi

echo "[1/4] Updating APT and installing system dependencies (OpenCV reqs)..."
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip libgl1 libglib2.0-0

echo "[2/4] Creating separate Python virtual environment for WSL (.venv_wsl)..."
python3 -m venv .venv_wsl

echo "[3/4] Activating virtual environment..."
source .venv_wsl/bin/activate

echo "[4/4] Installing Python requirements from ${REQ_FILE}..."
pip install -r "$REQ_FILE"

echo "=========================================="
echo " Setup completed successfully!"
echo " To activate this environment in the future, run:"
echo " source .venv_wsl/bin/activate"
echo " Profiles: --runtime-only | --training"
echo "=========================================="
