"""Model-size ablation: train YOLO11n and YOLO11s with hyperparameters identical to
the deployed YOLO11m run (runs/detect/pufferfish/args.yaml), then evaluate both
val and test splits. Produces benchmark_pc/ablation_sizes.json.

Rationale: an edge-AI paper must justify the chosen model scale. Reviewers expect a
speed/accuracy trade-off table across model capacities, not a single arbitrary choice.
"""
from __future__ import annotations
import json, time
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset" / "data.yaml"
OUT = ROOT / "benchmark_pc" / "ablation_sizes.json"

# Exactly the non-default hyperparameters used for the deployed YOLO11m run.
COMMON = dict(
    data=str(DATA), epochs=50, imgsz=640, batch=8, device=0, workers=2,
    optimizer="auto", lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
    warmup_epochs=3.0, box=7.5, cls=0.5, dfl=1.5,
    hsv_h=0.04, hsv_s=0.7, hsv_v=0.6, degrees=10.0, translate=0.1, scale=0.5,
    fliplr=0.5, bgr=0.1, mosaic=1.0, mixup=0.15, erasing=0.4, close_mosaic=10,
    auto_augment="randaugment", amp=True, seed=0, deterministic=True,
    patience=100, exist_ok=True, plots=True, val=True,
)

VARIANTS = [("yolo11n.pt", "pufferfish_n"), ("yolo11s.pt", "pufferfish_s")]


def evaluate(model: YOLO) -> dict:
    out = {}
    for split in ("val", "test"):
        r = model.val(data=str(DATA), split=split, imgsz=640, device=0,
                      verbose=False, plots=False)
        out[split] = {
            "precision": round(float(r.box.mp), 4),
            "recall": round(float(r.box.mr), 4),
            "mAP50": round(float(r.box.map50), 4),
            "mAP50_95": round(float(r.box.map), 4),
        }
    return out


def main() -> None:
    results = {}
    if OUT.exists():
        results = json.loads(OUT.read_text(encoding="utf-8"))

    for weights, run_name in VARIANTS:
        print(f"\n===== TRAINING {weights} -> {run_name} =====", flush=True)
        t0 = time.perf_counter()
        model = YOLO(weights)
        model.train(project=str(ROOT / "runs" / "detect"), name=run_name, **COMMON)
        train_s = time.perf_counter() - t0

        best = ROOT / "runs" / "detect" / run_name / "weights" / "best.pt"
        best_model = YOLO(str(best))
        metrics = evaluate(best_model)

        results[weights.replace(".pt", "")] = {
            "run_name": run_name,
            "weights": str(best),
            "train_time_sec": round(train_s, 1),
            "model_size_mb": round(best.stat().st_size / 1e6, 2),
            "metrics": metrics,
        }
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"DONE {weights}: {metrics}  ({train_s/60:.1f} min)", flush=True)

    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
