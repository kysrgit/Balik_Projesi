from ultralytics import YOLO
import os
from pathlib import Path

# --- CONFIGURATION ---
# We use YOLOv8s as requested, but YOLOv11s is a drop-in replacement if you change the model name.
MODEL_NAME = 'yolov8s.pt' 
DATA_CONFIG = Path(__file__).parent.parent / "dataset" / "data.yaml"
PROJECT_DIR = Path(__file__).parent.parent / "runs" / "detect"
EPOCHS = 50
IMG_SIZE = 640

def train():
    # 1. Load the model
    print(f"[INFO] Loading {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)

    # 2. Train with Underwater-Specific Augmentations
    # We tune HSV (Hue, Saturation, Value) to simulate underwater color shifts (blue/green tint).
    # We also ensure mosaic is on for robustness.
    print(f"[INFO] Starting training on {DATA_CONFIG}...")
    
    results = model.train(
        data=str(DATA_CONFIG),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        project=str(PROJECT_DIR),
        name='pufferfish_yolov8s',
        
        # --- Augmentation Hyperparameters for Underwater ---
        hsv_h=0.04,  # Increase hue shift (default 0.015) to handle water color var
        hsv_s=0.7,   # Saturation gain (default 0.7)
        hsv_v=0.4,   # Value gain (default 0.4) - lighting variation
        degrees=10,  # Small rotation
        translate=0.1, # Translation
        scale=0.5,   # Scaling (fish can be near or far)
        fliplr=0.5,  # Left-right flip (fish are symmetric)
        mosaic=1.0,  # Strong mosaic augmentation
        
        # --- System ---
        device='cpu', # Force CPU if no GPU available on dev machine, change to 0 if you have NVIDIA GPU
        workers=4,
        exist_ok=True
    )

    print(f"[INFO] Training complete. Best model saved to {PROJECT_DIR}/pufferfish_yolov8s/weights/best.pt")

if __name__ == "__main__":
    if not DATA_CONFIG.exists():
        print(f"[ERROR] Data config not found at {DATA_CONFIG}")
        print("Please run data_prep.py first!")
    else:
        train()
