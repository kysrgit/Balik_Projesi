#!/bin/bash

echo "--- Pufferfish System Installer for Raspberry Pi 5 ---"

# 1. Update System
echo "[1/3] Updating System..."
sudo apt-get update && sudo apt-get upgrade -y

# 2. Install System Dependencies (OpenCV requires these)
echo "[2/3] Installing System Libraries..."
sudo apt-get install -y libgl1-mesa-glx python3-opencv python3-pip

# 3. Install Python Libraries
echo "[3/3] Installing Python Packages..."
# Break system packages restriction if needed, or use a venv (recommended)
# For simplicity in this prototype, we use --break-system-packages (common on Pi 5 Bookworm)
pip3 install ultralytics onnxruntime gpiozero --break-system-packages

echo "--- Installation Complete! ---"
echo "To run the app:"
echo "python3 app/main_pi.py"
