#!/usr/bin/env python3
"""
Balon Baligi Tespit Sistemi
Ana calistirici - headless ve gui modlari destekler
"""
import argparse
import time
import os
import sys
import cv2

# Proje path ayari
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core import config, Camera, Detector, gpio
from app.utils import apply_clahe, draw_boxes


def save_detection(frame, boxes, confs, save_dir):
    """Tespit edilen frame'i kaydet"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(save_dir, f"fish_{timestamp}.jpg")
    
    result = draw_boxes(frame, boxes, confs)
    cv2.imwrite(path, result)
    print(f"Kaydedildi: {path}")
    return path


def run(use_clahe=True, show_gui=False):
    print("=" * 40)
    print("Balon Baligi Tespit Sistemi")
    print("=" * 40)
    
    # GPIO baslat
    gpio.init()
    
    # Detector yukle
    try:
        detector = Detector(config.MODEL_PATH)
    except Exception as e:
        print(f"Model yuklenemedi: {e}")
        return
    
    # Kamera baslat
    try:
        # Pi'de mi calisiyoruz kontrol et
        is_pi = os.path.exists('/sys/class/thermal/thermal_zone0/temp')
        cam = Camera(config.CAM_WIDTH, config.CAM_HEIGHT, use_pi=is_pi)
    except Exception as e:
        print(f"Kamera hatasi: {e}")
        return
    
    print("Izleme baslatildi (Ctrl+C ile durdur)")
    print("=" * 40)
    
    prev_time = time.time()
    last_save = 0
    frame_count = 0
    
    try:
        while True:
            frame = cam.read()
            if frame is None:
                time.sleep(0.1)
                continue
            
            frame_count += 1
            
            # CLAHE on isleme
            if use_clahe:
                processed = apply_clahe(frame, config.CLAHE_CLIP)
            else:
                processed = frame
            
            # Her N frame'de tespit yap (performans icin)
            boxes, confs = [], []
            if frame_count % config.SKIP_FRAMES == 0:
                boxes, confs = detector.detect(processed, config.CONF_THRESH)
            
            # Tespit varsa
            if len(boxes) > 0:
                gpio.on()
                
                # Saniyede max 1 kayit
                now = time.time()
                if now - last_save >= 1.0:
                    save_detection(processed, boxes, confs, str(config.DETECTION_DIR))
                    last_save = now
            else:
                gpio.off()
            
            # GUI modu
            if show_gui:
                display = draw_boxes(processed, boxes, confs)
                cv2.imshow("Tespit", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # FPS hesapla ve logla
            now = time.time()
            fps = 1.0 / (now - prev_time) if prev_time else 0
            prev_time = now
            
            if frame_count % 30 == 0:  # her saniye logla
                status = "TESPIT!" if len(boxes) > 0 else "Araniyor"
                print(f"FPS: {fps:.1f} | {status}")
    
    except KeyboardInterrupt:
        print("\nDurduruluyor...")
    finally:
        cam.release()
        gpio.off()
        if show_gui:
            cv2.destroyAllWindows()
        print("Sistem kapatildi.")


def main():
    parser = argparse.ArgumentParser(description="Balon Baligi Tespit Sistemi")
    parser.add_argument('--gui', action='store_true', help='GUI modunda calistir')
    parser.add_argument('--no-clahe', action='store_true', help='CLAHE on islemeyi kapat')
    args = parser.parse_args()
    
    run(use_clahe=not args.no_clahe, show_gui=args.gui)


if __name__ == "__main__":
    main()
