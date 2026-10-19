"""Faz 2 egitim deneyleri (E2): veri varyantlari, sabit tarif.

Tarif training/train_v2_improved.py ile ayni (mixup=0, scale=0.7, cos_lr) ve tum
deneylerde sabit; degisen yalnizca egitim verisidir (scripts/build_trainset.py).
Model secimi (best.pt) yalnizca val_v2 ile yapilir. Egitim bitince best.pt altin
sette (gold_v1 + bulanik kopya) scripts/eval_gold.py ile olculur.

Kullanim (GPU ortami):
    .venv_gpu/Scripts/python.exe training/train_e2.py --variants E2-0_base E2-1_hardneg E2-2_blur E2-3_full
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset_v2"
RESULTS = ROOT / "experiments" / "E2_training.json"

RECIPE = dict(
    epochs=60, patience=15, imgsz=640, batch=8, device=0, workers=2, optimizer="auto",
    lr0=0.01, lrf=0.01, cos_lr=True, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0,
    box=7.5, cls=0.5, dfl=1.5, hsv_h=0.04, hsv_s=0.7, hsv_v=0.6, degrees=10.0, translate=0.1,
    scale=0.7, fliplr=0.5, bgr=0.1, mosaic=1.0, close_mosaic=10, mixup=0.0,
    auto_augment="randaugment", erasing=0.4, amp=True, deterministic=True,
    exist_ok=True, plots=True, val=True,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true", help="best.pt varsa egitimi atla")
    args = ap.parse_args()
    from ultralytics import YOLO

    for v in args.variants:
        # yolo11s icin eski adlandirma korunur (E2 sonuclariyla uyumlu); diger modellerde model adi eklenir
        stem = Path(args.model).stem
        name = f"{v}_s{args.seed}" if stem == "yolo11s" else f"{v}_{stem}_s{args.seed}"
        best = ROOT / "runs" / "e2" / name / "weights" / "best.pt"
        if args.skip_existing and best.exists():
            print(f"===== SKIP {name} (zaten var) =====", flush=True)
            continue
        print(f"===== TRAIN {name} ({args.model}) =====", flush=True)
        t0 = time.perf_counter()
        YOLO(args.model).train(data=str(DATA_DIR / f"{v}.yaml"), seed=args.seed,
                               project=str(ROOT / "runs" / "e2"), name=name, **RECIPE)
        minutes = (time.perf_counter() - t0) / 60
        best = ROOT / "runs" / "e2" / name / "weights" / "best.pt"
        # altin sette olcum (ayri surecte: GPU bellegi bosalsin)
        out = f"experiments/E2_gold_{name}.json"
        subprocess.run([sys.executable, str(ROOT / "scripts" / "eval_gold.py"), "--models", str(best),
                        "--sets", "gold", "gold_blur", "gold_ext", "--out", out], check=True, cwd=ROOT)
        res = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
        res[name] = {"variant": v, "model": args.model, "seed": args.seed, "train_minutes": round(minutes, 1),
                     "weights": best.relative_to(ROOT).as_posix(), "gold_eval": out, "recipe": RECIPE}
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"DONE {name} in {minutes:.1f} min", flush=True)


if __name__ == "__main__":
    main()
