# Merkezi ayarlar
import os
from pathlib import Path

# Proje kök dizini
ROOT_DIR = Path(__file__).parent.parent.parent

# Model
MODEL_PATH = ROOT_DIR / "models" / "pufferfish_pi_int8.onnx"
CONF_THRESH = 0.60
DETECTOR_IMGSZ = 640 # Model ONNX olarak 640x640 boyutunda sabit (fixed) ihraç edildiği için değiştirilemez.

# GIS & Veritabanı
GPS_PORT = "/dev/ttyAMA0"
GPS_BAUDRATE = 9600
DB_PATH = ROOT_DIR / "spatial_log.sqlite"
MAX_MAP_POINTS = 5000
GPS_STALE_TIMEOUT = 10.0  # GPS verisinin geçerlilik süresi (saniye)

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
# CORS: Listen only on same-origin by default. Set ALLOWED_ORIGINS env var (comma-separated) for cross-origin access.
ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', None)
if ALLOWED_ORIGINS:
    if ',' in str(ALLOWED_ORIGINS):
        ALLOWED_ORIGINS = [o.strip() for o in ALLOWED_ORIGINS.split(',')]
    else:
        ALLOWED_ORIGINS = str(ALLOWED_ORIGINS).strip()

STREAM_FPS = 15
JPEG_QUALITY = 70
WS_STREAM_QUALITY = 50  # WebSocket base64 stream JPEG kalitesi
DASHBOARD_SAVE_INTERVAL = 1.0  # Max 1 detection log/save per second

# CLAHE
CLAHE_CLIP = 3.0
CLAHE_GRID = (8, 8)

# Asenkron Motor
EVENTLET_ENABLED = True
