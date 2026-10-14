"""Aday modelleri ayni protokolle karsilastirir: dogruluk + yanlis alarm.

Her model icin:
  - test ve valid split'te mAP@50, mAP@50-95, precision, recall (ultralytics val, batch=1)
  - valid split'teki balıksız (bos etiketli) goruntulerde conf>=0.60 yanlis alarm sayisi

Kullanim:
    python scripts/compare_models.py --models models/candidates/*.onnx --out benchmark_pc/model_compare.json
"""
from __future__ import annotations
import argparse, json, platform, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset" / "data.yaml"
CONF = 0.60


def negative_images() -> list[Path]:
    img_dir = ROOT / "dataset" / "valid" / "images"
    lbl_dir = ROOT / "dataset" / "valid" / "labels"
    out = []
    for p in sorted(img_dir.glob("*.jpg")):
        lbl = lbl_dir / f"{p.stem}.txt"
        if not lbl.exists() or not lbl.read_text(encoding="utf-8").strip():
            out.append(p)
    return out


def evaluate(model_path: Path, negs: list[Path], device: str) -> dict:
    from ultralytics import YOLO
    res = {"file_mb": round(model_path.stat().st_size / 1e6, 2)}
    with tempfile.TemporaryDirectory() as tmp:
        for split in ("test", "val"):
            m = YOLO(str(model_path), task="detect")
            r = m.val(data=str(DATA), split=split, imgsz=640, batch=1, device=device,
                      verbose=False, plots=False, project=tmp, name=split)
            res[split] = {"mAP50": round(r.box.map50, 4), "mAP50_95": round(r.box.map, 4),
                          "precision": round(r.box.mp, 4), "recall": round(r.box.mr, 4)}
    m = YOLO(str(model_path), task="detect")
    fp_img = fp_box = 0
    for p in negs:
        n = len(m.predict(source=str(p), conf=CONF, imgsz=640, device=device, verbose=False)[0].boxes)
        fp_img += bool(n)
        fp_box += n
    res["false_alarm"] = {"images": fp_img, "boxes": fp_box, "negatives": len(negs)}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="benchmark_pc/model_compare.json")
    args = ap.parse_args()

    import ultralytics, onnxruntime
    negs = negative_images()
    report = {"meta": {"ultralytics": ultralytics.__version__, "onnxruntime": onnxruntime.__version__,
                       "platform": platform.platform(), "device": args.device, "conf_false_alarm": CONF},
              "models": {}}
    for mp in map(Path, args.models):
        print(f"== {mp.name}")
        report["models"][mp.name] = r = evaluate(mp, negs, args.device)
        print(f"   test mAP50={r['test']['mAP50']:.4f} mAP50-95={r['test']['mAP50_95']:.4f} "
              f"R={r['test']['recall']:.4f} | yanlis alarm {r['false_alarm']['images']}/{len(negs)}")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
