#!/usr/bin/env python3
"""Model test"""
from ultralytics import YOLO

model = YOLO("models/pufferfish_pi_int8.onnx", task="detect")
print("Siniflar:", model.names)
