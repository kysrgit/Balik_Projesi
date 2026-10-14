"""
Yanlış Alarm (False Positive) Ölçüm Betiği.
Doğrulama kümesindeki etiketlenmemiş/boş etikete sahip (balıksız) 32 negatif görüntü üzerinde
modellerin hayali kutu (ghost box) üretme sıklığını karşılaştırır.
"""
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
VALID_IMG = ROOT / "dataset" / "valid" / "images"
VALID_LBL = ROOT / "dataset" / "valid" / "labels"

negs = [
    p for p in VALID_IMG.glob("*.jpg")
    if not (VALID_LBL / f"{p.stem}.txt").exists() or (VALID_LBL / f"{p.stem}.txt").read_text(encoding="utf-8").strip() == ""
]

print(f"Toplam negatif (balıksız) görüntü sayısı: {len(negs)}")

models_to_test = [
    ("YOLO11m (eski)", ROOT / "archive" / "models_v1" / "yolo11m_pufferfish.pt"),
    ("YOLO11s (yeni)", ROOT / "models" / "pufferfish_s_v2.pt"),
    ("YOLO11s FP32 ONNX", ROOT / "models" / "pufferfish_s_v2.onnx"),
    ("YOLO11s INT8 ONNX", ROOT / "models" / "pufferfish_s_v2_int8.onnx"),
]

for name, model_path in models_to_test:
    if not model_path.exists():
        print(f"[Atlandı] {name}: {model_path} bulunamadı")
        continue
    m = YOLO(str(model_path), task="detect")
    fp_img = 0
    fp_box = 0
    for p in negs:
        r = m.predict(source=str(p), conf=0.60, imgsz=640, verbose=False)
        n = len(r[0].boxes)
        if n > 0:
            fp_img += 1
            fp_box += n
    print(f"[{name}] Yanlış alarm veren görüntü: {fp_img}/{len(negs)} ({fp_img/len(negs)*100:.1f}%), Toplam hayali kutu: {fp_box}")
