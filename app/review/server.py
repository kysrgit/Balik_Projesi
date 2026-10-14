"""Tespit inceleme arayuzu (PC).

Etiketli veri seti veya kendi yukledigin goruntu/videolar uzerinde modelin ne
buldugunu gozle kontrol etmek icin. Etiket (ground truth) ile tahminler ust uste
cizilir; dogru tespit / yanlis alarm / kacirilan balik ayri renklerle gosterilir.

Calistirma (proje kokunden):
    python -m app.review.server          -> http://localhost:5050
"""
from __future__ import annotations

import base64
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from app.core import config
from app.utils.image import apply_clahe

ROOT = config.ROOT_DIR
DATASET = ROOT / "dataset"
SPLITS = {"test": "test", "valid": "valid", "train": "train"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PRED_CONF = 0.05          # dusuk esikle tahmin al; asil esik arayuzde kaydirgacla secilir
MAX_VIDEO_FRAMES = 120

app = Flask(__name__)
_models: dict[str, object] = {}
_cache: dict[tuple, dict] = {}
_lock = threading.Lock()


def list_models() -> list[str]:
    found = []
    for d in (ROOT / "models", ROOT / "models" / "candidates"):
        for p in sorted(d.glob("*")):
            if p.suffix in {".onnx", ".pt"} and not p.name.startswith("_"):
                found.append(p.relative_to(ROOT).as_posix())
    return found


def get_model(rel: str):
    if rel not in list_models():
        abort(400, "bilinmeyen model")
    if rel not in _models:
        from ultralytics import YOLO
        _models[rel] = YOLO(str(ROOT / rel), task="detect")
    return _models[rel]


def predict(model, frame: np.ndarray, clahe: bool) -> list[list[float]]:
    """Normalize [x1,y1,x2,y2,conf] listesi dondurur.

    Uygulamadaki Detector ile ayni on isleme: kare oldugu gibi verilir, ultralytics
    en-boy oranini koruyarak 640'a letterbox yapar; CLAHE istege bagli.
    """
    h, w = frame.shape[:2]
    img = apply_clahe(frame, clip=config.CLAHE_CLIP, grid=config.CLAHE_GRID) if clahe else frame
    r = model.predict(source=img, conf=PRED_CONF, imgsz=config.DETECTOR_IMGSZ, verbose=False)[0]
    out = []
    if r.boxes is not None and len(r.boxes):
        for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            out.append([round(float(x1 / w), 5), round(float(y1 / h), 5),
                        round(float(x2 / w), 5), round(float(y2 / h), 5), round(float(c), 4)])
    return out


def read_gt(label: Path) -> list[list[float]]:
    if not label.exists():
        return []
    boxes = []
    for line in label.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 5:  # YOLO kutu: sinif cx cy w h
            cx, cy, bw, bh = map(float, parts[1:5])
            boxes.append([round(cx - bw / 2, 5), round(cy - bh / 2, 5),
                          round(cx + bw / 2, 5), round(cy + bh / 2, 5)])
        elif len(parts) > 5:  # Roboflow poligonu: sinif x1 y1 x2 y2 ... -> sinirlayan kutu
            xs, ys = list(map(float, parts[1::2])), list(map(float, parts[2::2]))
            boxes.append([round(min(xs), 5), round(min(ys), 5), round(max(xs), 5), round(max(ys), 5)])
    return boxes


def options() -> tuple[str, bool]:
    data = request.get_json(silent=True) or request.form
    model = data.get("model") or config.MODEL_PATH.relative_to(ROOT).as_posix()
    clahe = str(data.get("clahe", "false")).lower() in {"1", "true", "on"}
    return model, clahe


@app.route("/")
def index():
    return render_template("review.html",
                           models=list_models(),
                           default_model=config.MODEL_PATH.relative_to(ROOT).as_posix(),
                           splits=[s for s in SPLITS if (DATASET / s / "images").exists()])


@app.route("/img/<split>/<path:name>")
def image(split, name):
    if split not in SPLITS:
        abort(404)
    return send_from_directory(DATASET / SPLITS[split] / "images", name)


@app.route("/api/run", methods=["POST"])
def run_split():
    model_rel, clahe = options()
    split = (request.get_json(silent=True) or {}).get("split", "test")
    if split not in SPLITS:
        abort(400, "bilinmeyen split")
    key = (model_rel, split, clahe)
    with _lock:
        if key not in _cache:
            model = get_model(model_rel)
            img_dir = DATASET / SPLITS[split] / "images"
            lbl_dir = DATASET / SPLITS[split] / "labels"
            items, t_total = [], 0.0
            for p in sorted(x for x in img_dir.iterdir() if x.suffix.lower() in IMG_EXT):
                frame = cv2.imread(str(p))
                if frame is None:
                    continue
                t0 = time.perf_counter()
                preds = predict(model, frame, clahe)
                t_total += time.perf_counter() - t0
                items.append({"name": p.name, "src": f"/img/{split}/{p.name}",
                              "gt": read_gt(lbl_dir / f"{p.stem}.txt"), "pred": preds})
            _cache[key] = {"items": items, "ms_per_image": round(1000 * t_total / max(len(items), 1), 1),
                           "labeled": True}
    return jsonify(_cache[key])


def _data_url(frame: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


@app.route("/api/upload", methods=["POST"])
def upload():
    model_rel, clahe = options()
    model = get_model(model_rel)
    items, t_total, n = [], 0.0, 0
    with _lock:
        for f in request.files.getlist("files"):
            suffix = Path(f.filename or "").suffix.lower()
            if suffix in IMG_EXT:
                frame = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
                frames = [(f.filename, frame)] if frame is not None else []
            else:  # video: saniyede ~1 kare ornekle
                with tempfile.NamedTemporaryFile(suffix=suffix or ".mp4", delete=False) as tmp:
                    f.save(tmp.name)
                cap = cv2.VideoCapture(tmp.name)
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                step = max(int(round(fps)), 1)
                if total and total / step > MAX_VIDEO_FRAMES:
                    step = total // MAX_VIDEO_FRAMES
                frames, idx = [], 0
                while len(frames) < MAX_VIDEO_FRAMES:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frames.append((f"{f.filename} @ {idx / fps:.1f}s", frame))
                    idx += step
                cap.release()
                Path(tmp.name).unlink(missing_ok=True)
            for name, frame in frames:
                t0 = time.perf_counter()
                preds = predict(model, frame, clahe)
                t_total += time.perf_counter() - t0
                n += 1
                items.append({"name": name, "src": _data_url(frame), "gt": None, "pred": preds})
    return jsonify({"items": items, "ms_per_image": round(1000 * t_total / max(n, 1), 1), "labeled": False})


def main():
    print("Inceleme arayuzu: http://localhost:5050")
    app.run(host="127.0.0.1", port=5050, debug=False, threaded=True)


if __name__ == "__main__":
    main()
