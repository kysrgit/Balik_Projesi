# YOLO11m egitim scripti
from ultralytics import YOLO
from pathlib import Path

DATA_YAML = Path(__file__).parent.parent / "dataset" / "data.yaml"
OUTPUT_DIR = Path(__file__).parent.parent / "runs" / "detect"

# Egitim parametreleri
EPOCHS = 50
IMG_SIZE = 640
MODEL = 'yolo11m.pt'


def train():
    if not DATA_YAML.exists():
        print(f"data.yaml yok: {DATA_YAML}")
        print("Once data_prep.py calistir!")
        return
    
    print(f"Egitim basliyor: {MODEL}")
    model = YOLO(MODEL)
    
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        project=str(OUTPUT_DIR),
        name='pufferfish',
        
        # Sualti augmentasyon
        hsv_h=0.04,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        
        device=0,
        exist_ok=True
    )
    
    print(f"Bitti! Model: {OUTPUT_DIR}/pufferfish/weights/best.pt")


if __name__ == "__main__":
    train()
