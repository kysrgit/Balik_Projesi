#!/usr/bin/env python3
"""Kamera testi"""
import cv2
import time

print("=== Kamera Testi ===")

for src in [0, 7, 20]:
    print(f"Deneniyor: /dev/video{src}...")
    cap = cv2.VideoCapture(src)
    
    if cap.isOpened():
        time.sleep(0.5)
        ret, frame = cap.read()
        if ret:
            print(f"  OK - {frame.shape}")
            cv2.imwrite(f"/tmp/test_{src}.jpg", frame)
        else:
            print(f"  Frame alinamadi")
        cap.release()
    else:
        print(f"  Acilamadi")

print("=== Bitti ===")
