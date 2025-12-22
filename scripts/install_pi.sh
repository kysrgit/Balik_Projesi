#!/bin/bash
echo "=== Pi Kurulum ==="

sudo apt update && sudo apt upgrade -y
sudo apt install -y libgl1-mesa-glx python3-opencv python3-pip

pip3 install ultralytics onnxruntime gpiozero --break-system-packages

echo "=== Bitti ==="
echo "python3 app/main.py"
