"""Altin test setinde (gold_v1) model degerlendirmesi — ADR-000 olcutleriyle.

Her model icin (ayni protokol):
  - recall, precision, F1 (IoU>=0.5, guvene gore aç gozlu eslesme) secilen esiklerde
  - bootstrap %95 guven araligi (goruntu duzeyinde yeniden ornekleme, 2000 tekrar)
  - kucuk nesne recall'i (kareye gore alan < %1, ADR-000 K2)
  - negatif goruntulerde alarm orani (ADR-000 K4)
  - ayni olculer hareket bulanikligi kopyasinda (gold_v1_blur, ADR-000 K7)
  - mAP@50 ve mAP@50-95 (ultralytics val)

Kullanim:
    python scripts/eval_gold.py --models models/pufferfish_s_v2_int8.onnx models/candidates/cand_n_int8.onnx \
        --out experiments/E1a_gold_baseline.json
"""
from __future__ import annotations
import argparse, json, platform, tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "dataset_gold_v1"
GOLD_BLUR = ROOT / "dataset_gold_v1_blur"
THRESHOLDS = (0.25, 0.4, 0.5, 0.6)


def load_gt(lbl: Path) -> list[list[float]]:
    out = []
    for line in lbl.read_text(encoding="utf-8").splitlines():
        cx, cy, w, h = map(float, line.split()[1:5])
        out.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return out


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def match(gt, preds, thr):
    """Goruntu basina (tp, fp, fn, kucuk_toplam, kucuk_bulunan)."""
    preds = sorted([p for p in preds if p[4] >= thr], key=lambda p: -p[4])
    used, tp = set(), 0
    for p in preds:
        best, bi = -1, 0.5
        for i, g in enumerate(gt):
            if i not in used:
                v = iou(p, g)
                if v >= bi:
                    best, bi = i, v
        if best >= 0:
            used.add(best); tp += 1
    small = [i for i, g in enumerate(gt) if (g[2] - g[0]) * (g[3] - g[1]) < 0.01]
    return tp, len(preds) - tp, len(gt) - tp, len(small), sum(1 for i in small if i in used)


def predict_all(model_path: Path, root: Path) -> list[dict]:
    from ultralytics import YOLO
    m = YOLO(str(model_path), task="detect")
    items = []
    for img in sorted((root / "images").glob("*.jpg")):
        r = m.predict(source=str(img), conf=0.05, imgsz=640, verbose=False)[0]
        h, w = r.orig_shape
        preds = [[x1 / w, y1 / h, x2 / w, y2 / h, float(c)]
                 for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy().tolist(), r.boxes.conf.cpu().numpy())]
        items.append({"gt": load_gt(root / "labels" / f"{img.stem}.txt"), "pred": preds})
    return items


def metrics(items, thr, rng, n_boot=2000) -> dict:
    per = np.array([match(it["gt"], it["pred"], thr) for it in items], dtype=float)  # tp fp fn small smallfound
    neg = np.array([len(it["gt"]) == 0 for it in items])
    alarm = np.array([any(p[4] >= thr for p in it["pred"]) for it in items])

    def agg(rows, negm, alm):
        tp, fp, fn, sm, smf = rows.sum(axis=0)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return {"recall": r, "precision": p, "f1": 2 * p * r / (p + r) if p + r else 0.0,
                "small_recall": smf / sm if sm else float("nan"),
                "neg_alarm_rate": alm[negm].mean() if negm.any() else float("nan")}

    point = agg(per, neg, alarm)
    boots = {k: [] for k in point}
    n = len(items)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        b = agg(per[idx], neg[idx], alarm[idx])
        for k, v in b.items():
            boots[k].append(v)
    out = {}
    for k, v in point.items():
        arr = np.array(boots[k], dtype=float)
        arr = arr[~np.isnan(arr)]
        lo, hi = (np.percentile(arr, [2.5, 97.5]) if len(arr) else (float("nan"), float("nan")))
        out[k] = {"value": round(float(v), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}
    tp, fp, fn, sm, smf = per.sum(axis=0)
    out["counts"] = {"tp": int(tp), "fp": int(fp), "fn": int(fn), "small_gt": int(sm),
                     "neg_images": int(neg.sum()), "neg_alarms": int((alarm & neg).sum())}
    return out


def map_scores(model_path: Path, root: Path) -> dict:
    from ultralytics import YOLO
    with tempfile.TemporaryDirectory() as tmp:
        r = YOLO(str(model_path), task="detect").val(data=str(root / "data.yaml"), split="test", imgsz=640,
                                                     batch=1, device="cpu", verbose=False, plots=False, project=tmp)
    return {"mAP50": round(float(r.box.map50), 4), "mAP50_95": round(float(r.box.map), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", default="experiments/E1a_gold_baseline.json")
    args = ap.parse_args()
    import ultralytics, onnxruntime
    rng = np.random.default_rng(2026)
    report = {"meta": {"gold": "gold_v1", "images": len(list((GOLD / "images").glob("*.jpg"))),
                       "ultralytics": ultralytics.__version__, "onnxruntime": onnxruntime.__version__,
                       "platform": platform.platform(), "iou": 0.5, "bootstrap": 2000}, "models": {}}
    for mp in map(Path, args.models):
        print(f"== {mp.name}")
        res = {"file_mb": round(mp.stat().st_size / 1e6, 2) if mp.is_file() else None}
        for name, root in (("gold", GOLD), ("gold_blur", GOLD_BLUR)):
            items = predict_all(mp, root)
            res[name] = {f"conf_{t}": metrics(items, t, rng) for t in THRESHOLDS}
            res[name].update(map_scores(mp, root))
            m6 = res[name]["conf_0.6"]
            print(f"   {name:9s} @0.6 R={m6['recall']['value']:.3f} {m6['recall']['ci95']} P={m6['precision']['value']:.3f} "
                  f"F1={m6['f1']['value']:.3f} kucukR={m6['small_recall']['value']:.3f} negAlarm={m6['neg_alarm_rate']['value']:.3f} "
                  f"mAP50={res[name]['mAP50']}")
        report["models"][mp.name] = res
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
