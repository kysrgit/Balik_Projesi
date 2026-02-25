import sys
import os
import cv2
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core import config, Camera, Detector

print("1. Kamera Baslatiliyor...")
cam = Camera(640, 480, use_pi=False)
frame = cam.read()

if frame is None:
    print("HATA: Kamera (veya Mock) frame okuyamadi!")
    sys.exit(1)

print(f"Frame okundu: {frame.shape}")

print("2. Model Yukleniyor...")
detector = Detector(config.MODEL_PATH)

print("3. Algilama Testi Yapiliyor...")
start = time.time()
boxes, confs = detector.detect(frame, conf=0.1)  # Dusuk threshold
print(f"Sure: {time.time() - start:.2f} saniye")

if boxes:
    print(f"BASARILI: {len(boxes)} adet nesne tespit edildi!")
    for b, c in zip(boxes, confs):
        print(f" - Kutu: {b}, Guven: {c:.2f}")
else:
    print("UYARI: Model calisti ama ekranda hicbir sey tespit edemedi.")

print("Test bitti.")
