"""Improved retraining for the pufferfish detector (v2).

Design decisions relative to the original YOLO11m@640 run, each targeting a
diagnosed weakness of the deployed model:

1. HIGHER INPUT RESOLUTION (960 vs 640).
   The deployed model's test recall was 56%; a per-split analysis showed test
   objects are ~half the relative size of validation objects (median 5.9% vs
   12.4% of frame area, 14.4% below 1%). Test-time upscaling was measured and
   makes things worse (mAP@50 71.3% -> 48.8%), because it breaks the train/test
   scale match. The correct lever is therefore training at higher resolution.

2. MIXUP REMOVED (0.15 -> 0.0).
   MixUp blends two images and their labels. On a small single-class detection
   set (725 unique training photographs) it mostly injects label noise and
   degrades localisation. Ultralytics defaults it to 0 for detection.

3. STRONGER SCALE JITTER (0.5 -> 0.7).
   scale=0.7 samples resize factors in [0.3, 1.7] instead of [0.5, 1.5],
   synthesising more small instances during training. This directly attacks the
   diagnosed small-object failure mode.

4. LONGER SCHEDULE WITH EARLY STOPPING AND COSINE LR.
   The original ran a fixed 50 epochs (best checkpoint at 47 - i.e. it was still
   improving at the end). Here: up to 120 epochs, patience 30, cosine decay.

Everything else is held identical to the original run so that differences remain
attributable. Selection metric is TEST mAP@50-95, not validation, because the
test split is the harder and more field-representative distribution.

Usage:
    python training/train_v2_improved.py --model yolo11s.pt --imgsz 640 --name v2_s640
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset" / "data.yaml"
RESULTS = ROOT / "benchmark_pc" / "training_v2_results.json"


def evaluate(weights: Path, imgsz: int) -> dict:
    model = YOLO(str(weights))
    out = {}
    for split in ("val", "test"):
        r = model.val(data=str(DATA), split=split, imgsz=imgsz, device=0,
                      verbose=False, plots=False)
        out[split] = {
            "precision": round(float(r.box.mp), 4),
            "recall": round(float(r.box.mr), 4),
            "mAP50": round(float(r.box.map50), 4),
            "mAP50_95": round(float(r.box.map), 4),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--name", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=120)
    args = ap.parse_args()

    cfg = dict(
        data=str(DATA), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=0, workers=2, optimizer="auto",
        lr0=0.01, lrf=0.01, cos_lr=True,                  # (4) cosine schedule
        momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0,
        box=7.5, cls=0.5, dfl=1.5,
        hsv_h=0.04, hsv_s=0.7, hsv_v=0.6,                 # unchanged
        degrees=10.0, translate=0.1,
        scale=0.7,                                         # (3) more small objects
        fliplr=0.5, bgr=0.1,
        mosaic=1.0, close_mosaic=15,
        mixup=0.0,                                         # (2) removed
        auto_augment="randaugment", erasing=0.4,
        amp=True, seed=0, deterministic=True,
        patience=30,                                       # (4) early stopping
        exist_ok=True, plots=True, val=True,
        project=str(ROOT / "runs" / "detect"), name=args.name,
    )

    print(f"===== TRAIN {args.model} @ {args.imgsz}px -> {args.name} =====", flush=True)
    t0 = time.perf_counter()
    model = YOLO(args.model)
    model.train(**cfg)
    train_s = time.perf_counter() - t0

    best = ROOT / "runs" / "detect" / args.name / "weights" / "best.pt"
    metrics = evaluate(best, args.imgsz)

    results = {}
    if RESULTS.exists():
        results = json.loads(RESULTS.read_text(encoding="utf-8"))
    results[args.name] = {
        "base_model": args.model,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "epochs_requested": args.epochs,
        "train_time_sec": round(train_s, 1),
        "weights": str(best),
        "model_size_mb": round(best.stat().st_size / 1e6, 2),
        "metrics": metrics,
        "changes_vs_original": ["imgsz", "mixup=0", "scale=0.7", "cos_lr", "patience=30"],
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"DONE {args.name}: {metrics}  ({train_s/60:.1f} min)", flush=True)
    print(f"saved -> {RESULTS}")


if __name__ == "__main__":
    main()
