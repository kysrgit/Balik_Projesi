# Merkezi ayarlar
import os
from pathlib import Path

# Proje kök dizini
ROOT_DIR = Path(__file__).parent.parent.parent

# Model
MODEL_PATH = ROOT_DIR / "models" / "pufferfish_pi_int8.onnx"
CONF_THRESH = 0.60
DETECTOR_IMGSZ = 640 # Model ONNX olarak 640x640 boyutunda sabit (fixed) ihraç edildiği için değiştirilemez. 

# Kamera
CAM_WIDTH = 640
CAM_HEIGHT = 480
TARGET_FPS = 30
SKIP_FRAMES = 5  # her N frame'de tespit yap

# Kayit
DETECTION_DIR = ROOT_DIR / "detections"
THUMB_DIR = DETECTION_DIR / "thumbs"

# Dashboard
DASHBOARD_PORT = 5000
STREAM_FPS = 15
JPEG_QUALITY = 70

# CLAHE
CLAHE_CLIP = 3.0
CLAHE_GRID = (8, 8)
