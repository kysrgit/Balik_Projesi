#!/bin/bash

echo "--- Pufferfish System Launcher ---"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Check if venv exists, if so activate it
if [ -d ".venv_pi" ]; then
    echo "Activating virtual environment (.venv_pi)..."
    source .venv_pi/bin/activate
elif [ -d ".venv" ]; then
    echo "Activating virtual environment (.venv)..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

echo "Starting Application..."
python3 app/main.py
