"""
Local PC Benchmark Betiği (CPU Ölçümü) - YOLO11s Balon Balığı Modelleri.
PyTorch (.pt), ONNX FP32 ve ONNX INT8 modellerinin performans (gecikme, FPS, Precision, Recall, F1) metriklerini ölçer.
"""
from __future__ import annotations
import argparse, json, time, sys, platform
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.utils.image import apply_clahe

DATASET = ROOT / "dataset"
MODELS = {
    "PyTorch (.pt)":    ROOT / "models" / "pufferfish_s_v2.pt",
    "ONNX FP32":        ROOT / "models" / "pufferfish_s_v2.onnx",
    "ONNX INT8 (s_v2)": ROOT / "models" / "pufferfish_s_v2_int8.onnx",
}
EVAL_CONF = 0.60
BASE_CONF = 0.05
IOU_THR = 0.50
WARMUP = 15


def list_images(split: str) -> list[Path]:
    d = DATASET / split / "images"
    paths: list[Path] = []
    for pat in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        paths.extend(d.glob(pat))
    return sorted(paths)


def read_label_boxes(label_path: Path, w: int, h: int):
    if not label_path.exists():
        return []
    content = label_path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    boxes = []
    for line in content.splitlines():
        if not line.strip():
            continue
        vals = [float(v) for v in line.split()[1:]]
        if len(vals) == 4:
            xc, yc, bw, bh = vals
            boxes.append(((xc-bw/2)*w, (yc-bh/2)*h, (xc+bw/2)*w, (yc+bh/2)*h))
        else:
            xs, ys = vals[0::2], vals[1::2]
            boxes.append((min(xs)*w, min(ys)*h, max(xs)*w, max(ys)*h))
    return boxes


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter = iw*ih
    if inter <= 0:
        return 0.0
    ua = max(0.0, a[2]-a[0])*max(0.0, a[3]-a[1]) + max(0.0, b[2]-b[0])*max(0.0, b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


def collect(model_path: Path, split: str, device, use_clahe=True):
    from ultralytics import YOLO
    model = YOLO(str(model_path), task="detect")
    paths = list_images(split)
    w0 = cv2.imread(str(paths[0]))
    w0 = cv2.resize(w0, (640, 640))
    if use_clahe:
        w0 = apply_clahe(w0, clip=3.0)
    for _ in range(WARMUP):
        model.predict(source=w0, conf=BASE_CONF, imgsz=640, device=device, verbose=False)
    records = []
    for p in paths:
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        gt = read_label_boxes(DATASET/split/"labels"/f"{p.stem}.txt", w, h)
        img = cv2.resize(frame, (640, 640))
        if use_clahe:
            img = apply_clahe(img, clip=3.0)
        t0 = time.perf_counter()
        res = model.predict(source=img, conf=BASE_CONF, imgsz=640, device=device, verbose=False)
        lat = (time.perf_counter()-t0)*1000.0
        sx, sy = w/640.0, h/640.0
        preds = []
        for box in res[0].boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0].cpu().numpy())
            preds.append((x1*sx, y1*sy, x2*sx, y2*sy, float(box.conf)))
        records.append({"gt": gt, "pred": preds, "lat": lat})
    return records


def evaluate(records, thr=EVAL_CONF):
    tp = fp = fn = 0
    lats = [r["lat"] for r in records]
    for r in records:
        gts = list(r["gt"])
        preds = sorted([p for p in r["pred"] if p[4] >= thr], key=lambda x: x[4], reverse=True)
        matched = [False]*len(gts)
        for pr in preds:
            best, bi = 0.0, -1
            for i, g in enumerate(gts):
                if matched[i]:
                    continue
                cur = iou(pr[:4], g)
                if cur > best:
                    best, bi = cur, i
            if best >= IOU_THR and bi >= 0:
                matched[bi] = True; tp += 1
            else:
                fp += 1
        fn += matched.count(False)
    prec = tp/(tp+fp) if (tp+fp) else 0.0
    rec = tp/(tp+fn) if (tp+fn) else 0.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    mean_lat = float(np.mean(lats)) if lats else 0.0
    return {
        "images": len(records), "tp": tp, "fp": fp, "fn": fn,
        "precision": prec, "recall": rec, "f1": f1,
        "mean_latency_ms": mean_lat,
        "median_latency_ms": float(np.median(lats)) if lats else 0.0,
        "p90_latency_ms": float(np.percentile(lats, 90)) if lats else 0.0,
        "fps": 1000.0/mean_lat if mean_lat > 0 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", help="cpu or cuda (or 0)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["valid", "test"])
    args = ap.parse_args()

    device = 0 if args.device in ("cuda", "gpu", "0") else "cpu"

    meta = {"device_arg": args.device, "resolved_device": str(device),
            "python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch
        meta["torch"] = torch.__version__
        meta["cuda_available"] = bool(torch.cuda.is_available())
        meta["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as e:
        meta["torch_error"] = str(e)
    try:
        import onnxruntime as ort
        meta["onnxruntime"] = ort.__version__
        meta["ort_providers"] = ort.get_available_providers()
    except Exception as e:
        meta["ort_error"] = str(e)

    results = {"meta": meta, "eval_conf": EVAL_CONF, "iou": IOU_THR, "warmup": WARMUP, "runs": {}}
    for label, mp in MODELS.items():
        if not mp.exists():
            print(f"[skip] {label}: missing {mp}")
            continue
        results["runs"][label] = {"file_mb": mp.stat().st_size/1e6, "provider_used": None, "splits": {}}
        for split in args.splits:
            try:
                recs = collect(mp, split, device, use_clahe=True)
                try:
                    from ultralytics import YOLO
                    mm = YOLO(str(mp), task="detect")
                    _ = mm.predict(source=np.zeros((640,640,3), np.uint8), conf=0.9, imgsz=640, device=device, verbose=False)
                    sess = getattr(getattr(mm, "predictor", None), "model", None)
                    prov = None
                    if sess is not None and hasattr(sess, "session"):
                        prov = sess.session.get_providers()
                    results["runs"][label]["provider_used"] = prov
                except Exception:
                    pass
                ev = evaluate(recs, EVAL_CONF)
                results["runs"][label]["splits"][split] = ev
                print(f"[{args.device}] {label:16s} {split:5s} -> "
                      f"P={ev['precision']*100:5.2f}% R={ev['recall']*100:5.2f}% F1={ev['f1']*100:5.2f}% | "
                      f"{ev['mean_latency_ms']:7.2f} ms  {ev['fps']:6.2f} FPS  (n={ev['images']})")
            except Exception as e:
                results["runs"][label]["splits"][split] = {"error": str(e)}
                print(f"[{args.device}] {label} {split} FAILED: {e}")

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved -> {outp}")


if __name__ == "__main__":
    main()
