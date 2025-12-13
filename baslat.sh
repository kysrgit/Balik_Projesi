#!/bin/bash

echo "--- Pufferfish System Launcher ---"
cd ~/Balik_Projesi

# Check if venv exists, if so activate it
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

echo "Starting Application..."
python3 app/main_pi.py
