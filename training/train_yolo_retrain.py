# Fugaeus (Balon Balığı) Veri Seti YOLO11 Yeniden Eğitim Betiği
# Bu betik, dataset/data.yaml konfigürasyonundaki balon balığı verilerini YOLO11m modeli ile yeniden eğitir
# ve test/doğrulama metriklerini detaylı olarak kaydeder.

import os
import json
import time
from pathlib import Path
import torch
from ultralytics import YOLO

# Veri seti ve çıktı yolları
BASE_DIR = Path(__file__).parent.parent
DATA_YAML = BASE_DIR / "dataset" / "data.yaml"
OUTPUT_DIR = BASE_DIR / "runs" / "detect"
METRICS_JSON = BASE_DIR / "benchmark_pc" / "yolo_retrain_metrics.json"

# Eğitim hiperparametreleri
EPOCHS = 5
IMG_SIZE = 640
MODEL_NAME = "yolo11m.pt"
BATCH_SIZE = 8

def retrain_yolo():
    """
    Fugaeus balon balığı veri setini YOLO11m ile eğitir ve metrikleri raporlar.
    """
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Veri seti dosyası bulunamadı: {DATA_YAML}")

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[+] YOLO Eğitim Başlatılıyor... Cihaz: {device}")
    
    # YOLO Model Yükleme
    model = YOLO(MODEL_NAME)
    
    start_time = time.time()
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        project=str(OUTPUT_DIR),
        name="pufferfish_retrain",
        device=device,
        batch=BATCH_SIZE,
        workers=2,
        exist_ok=True,
        hsv_h=0.04,
        hsv_s=0.7,
        hsv_v=0.6,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        bgr=0.1
    )
    total_training_time = time.time() - start_time
    print(f"[+] YOLO Eğitimi Tamamlandı. Süre: {total_training_time:.2f} saniye")
    
    # Model Doğrulama (Validation Evaluation)
    print("[+] Model Doğrulama Değerlendirmesi Yapılıyor...")
    best_model_path = OUTPUT_DIR / "pufferfish_retrain" / "weights" / "best.pt"
    if not best_model_path.exists():
        best_model_path = BASE_DIR / "yolo11m.pt"
        
    trained_model = YOLO(str(best_model_path))
    val_metrics = trained_model.val(data=str(DATA_YAML), split="val")
    
    # Test Seti Değerlendirmesi
    print("[+] Test Seti Değerlendirmesi Yapılıyor...")
    test_metrics = trained_model.val(data=str(DATA_YAML), split="test")
    
    # Çıkarım Hızı (Latency Benchmark)
    dummy_img = torch.zeros((1, 3, IMG_SIZE, IMG_SIZE), device=device if torch.cuda.is_available() else "cpu")
    # Warmup
    for _ in range(10):
        _ = trained_model(dummy_img, verbose=False)
        
    t0 = time.time()
    n_runs = 50
    for _ in range(n_runs):
        _ = trained_model(dummy_img, verbose=False)
    avg_latency_ms = ((time.time() - t0) / n_runs) * 1000.0
    
    # Metrikleri hazırlama
    metrics_summary = {
        "model": "YOLO11m",
        "epochs": EPOCHS,
        "img_size": IMG_SIZE,
        "device": str(device),
        "total_training_time_sec": round(total_training_time, 2),
        "val_metrics": {
            "precision": float(val_metrics.results_dict.get("metrics/precision(B)", 0.0)),
            "recall": float(val_metrics.results_dict.get("metrics/recall(B)", 0.0)),
            "mAP50": float(val_metrics.results_dict.get("metrics/mAP50(B)", 0.0)),
            "mAP50_95": float(val_metrics.results_dict.get("metrics/mAP50-95(B)", 0.0))
        },
        "test_metrics": {
            "precision": float(test_metrics.results_dict.get("metrics/precision(B)", 0.0)),
            "recall": float(test_metrics.results_dict.get("metrics/recall(B)", 0.0)),
            "mAP50": float(test_metrics.results_dict.get("metrics/mAP50(B)", 0.0)),
            "mAP50_95": float(test_metrics.results_dict.get("metrics/mAP50-95(B)", 0.0))
        },
        "latency_ms_per_frame": round(avg_latency_ms, 2),
        "model_size_mb": round(os.path.getsize(str(best_model_path)) / (1024 * 1024), 2)
    }
    
    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4, ensure_ascii=False)
        
    print(f"[+] Metrikler kaydedildi: {METRICS_JSON}")
    return metrics_summary

if __name__ == "__main__":
    retrain_yolo()
