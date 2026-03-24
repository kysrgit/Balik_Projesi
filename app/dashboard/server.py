#!/usr/bin/env python3
"""
Dashboard Web Server
Flask + SocketIO ile gercek zamanli izleme paneli
"""
import eventlet

eventlet.monkey_patch()

import hmac
import json
import os
import queue
import shutil
import subprocess
import sys
import time
from datetime import datetime
from functools import wraps
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from flask_socketio import SocketIO, emit

# Path ayari
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.core import Camera, Detector, config
from app.core.detection_store import (
    append_detection_rows,
    build_detection_rows,
    ensure_csv_header,
    ensure_directory,
    save_image_with_retention,
    save_thumbnail,
)
from app.core.gps import gps_reader_thread, gps_state
from app.dashboard.stream import FrameBuffer, generate_mjpeg, get_base64_frame
from app.db.spatial import init_db, insert_detection
from app.export import WebhookNotifier, to_csv_download, to_darwincore_archive, to_geojson
from app.utils import draw_boxes
from app.utils.image import apply_clahe

# Flask app
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", os.urandom(32).hex())
socketio = SocketIO(app, cors_allowed_origins=config.ALLOWED_ORIGINS, async_mode="eventlet")

# Global state initialized from config
buffer = FrameBuffer()
conf_thresh = config.CONF_THRESH
clahe_clip = config.CLAHE_CLIP
overlay_clahe = False
overlay_boxes = True
is_recording = False

# Frame Queue for decoupled inference
frame_queue = queue.Queue(maxsize=5)

# Init CSV logger
CSV_LOG_FILE = str(config.CSV_LOG_FILE)
csv_log_queue = queue.Queue()

# Export / notifications
webhook_notifier = WebhookNotifier(rate_limit_seconds=60)

STREAM_TYPES = {"raw", "clahe", "detection"}
SECURITY_CSP = (
    "default-src 'self'; "
    "img-src 'self' data: https://*.basemaps.cartocdn.com https://cdnjs.cloudflare.com https://unpkg.com; "
    "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.tailwindcss.com https://cdnjs.cloudflare.com https://unpkg.com; "
    "style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com https://unpkg.com; "
    "font-src 'self' data: https://cdnjs.cloudflare.com; "
    "connect-src 'self' ws: wss:; "
    "frame-ancestors 'none'; "
    "base-uri 'self';"
)


def _token_valid() -> bool:
    if not config.API_TOKEN:
        return True

    header_token = request.headers.get("X-API-Token", "")
    auth_header = request.headers.get("Authorization", "")
    bearer_token = ""
    if auth_header.startswith("Bearer "):
        bearer_token = auth_header[7:]

    candidate = header_token or bearer_token
    return bool(candidate) and hmac.compare_digest(candidate, config.API_TOKEN)


def require_api_token(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not _token_valid():
            return jsonify({"status": "error", "message": "unauthorized"}), 401
        return view(*args, **kwargs)

    return wrapper


@app.after_request
def apply_security_headers(response):
    response.headers.setdefault("Content-Security-Policy", SECURITY_CSP)
    response.headers.setdefault("Referrer-Policy", "same-origin")
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    return response


def csv_logger_thread():
    ensure_csv_header(CSV_LOG_FILE)
    while True:
        row = csv_log_queue.get()
        append_detection_rows(CSV_LOG_FILE, [row])


def _read_cpu_temp():
    temp_path = Path("/sys/class/thermal/thermal_zone0/temp")
    if not temp_path.exists():
        return 0.0
    return int(temp_path.read_text(encoding="utf-8").strip()) / 1000


def _read_throttle_state():
    vcgencmd = shutil.which("vcgencmd")
    if not vcgencmd:
        return False
    result = subprocess.run(
        [vcgencmd, "get_throttled"],
        capture_output=True,
        text=True,
        timeout=1,
        check=False,
    )
    if result.returncode != 0 or "=" not in result.stdout:
        return False
    try:
        value = int(result.stdout.strip().split("=")[1], 16)
    except ValueError:
        return False
    return value != 0


def _read_fan_rpm():
    for path in Path("/sys/class/hwmon").glob("hwmon*/fan1_input"):
        try:
            return int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
    return None


# -- Sistem bilgileri --
def get_stats():
    import eventlet.tpool

    stats = {"cpu_temp": 0.0, "throttled": False, "fan_rpm": None}
    try:
        stats["cpu_temp"] = eventlet.tpool.execute(_read_cpu_temp)
    except Exception:
        pass

    try:
        stats["throttled"] = eventlet.tpool.execute(_read_throttle_state)
    except Exception:
        pass

    try:
        stats["fan_rpm"] = eventlet.tpool.execute(_read_fan_rpm)
    except Exception:
        pass

    return stats


# -- Routes --
@app.route("/")
def index():
    return render_template("index.html", token_required=bool(config.API_TOKEN))


@app.route("/video/<stream_type>")
def video(stream_type):
    if stream_type not in STREAM_TYPES:
        return jsonify({"status": "error", "message": "invalid stream type"}), 404
    return Response(
        generate_mjpeg(buffer, stream_type),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    global conf_thresh, clahe_clip, is_recording, overlay_clahe, overlay_boxes

    if request.method == "POST":
        if not _token_valid():
            return jsonify({"status": "error", "message": "unauthorized"}), 401
        data = request.get_json(silent=True) or {}
        if "confidence" in data:
            conf_thresh = float(data["confidence"])
        if "clahe_clip" in data:
            clahe_clip = float(data["clahe_clip"])
        if "overlay_clahe" in data:
            overlay_clahe = bool(data["overlay_clahe"])
        if "overlay_boxes" in data:
            overlay_boxes = bool(data["overlay_boxes"])
        return jsonify({"status": "ok"})

    return jsonify(
        {
            "confidence": conf_thresh,
            "clahe_clip": clahe_clip,
            "recording": is_recording,
            "overlay_clahe": overlay_clahe,
            "overlay_boxes": overlay_boxes,
            "token_required": bool(config.API_TOKEN),
        }
    )


@app.route("/api/snapshot", methods=["POST"])
@require_api_token
def snapshot():
    frame = buffer.get("detection")
    if frame is None:
        frame = buffer.get("raw")
    if frame is not None:
        snapshot_dir = config.DETECTION_DIR / "snapshots"
        name = f"snap_{datetime.now().strftime('%H%M%S')}.jpg"
        save_image_with_retention(
            frame,
            snapshot_dir,
            name,
            max_files=config.MAX_SNAPSHOT_FILES,
        )
        return jsonify({"status": "ok", "file": f"snapshots/{name}"})
    return jsonify({"status": "error"})


@app.route("/api/record", methods=["POST"])
@require_api_token
def record_toggle():
    global is_recording
    is_recording = not is_recording
    return jsonify({"status": "ok", "recording": is_recording})


@app.route("/detections/<path:name>")
def serve_file(name):
    return send_from_directory(str(config.DETECTION_DIR), name)


@app.route("/api/export/geojson")
def export_geojson():
    """GeoJSON export — harita servisleri, QGIS, Leaflet uyumlu"""
    data = to_geojson(CSV_LOG_FILE, str(config.DB_PATH))
    return Response(
        json.dumps(data, ensure_ascii=False, indent=2),
        mimetype="application/geo+json",
        headers={"Content-Disposition": "attachment; filename=pufferfish_detections.geojson"},
    )


@app.route("/api/export/csv")
def export_csv():
    """CSV download — araştırmacılar için"""
    csv_data = to_csv_download(CSV_LOG_FILE, str(config.DB_PATH))
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=pufferfish_detections.csv"},
    )


@app.route("/api/export/darwincore")
def export_darwincore():
    """DarwinCore Archive (ZIP) — GBIF / OBIS uyumlu"""
    zip_data = to_darwincore_archive(CSV_LOG_FILE, str(config.DB_PATH))
    return Response(
        zip_data,
        mimetype="application/zip",
        headers={"Content-Disposition": "attachment; filename=pufferfish_dwca.zip"},
    )


@app.route("/api/webhooks", methods=["GET", "POST", "DELETE"])
@require_api_token
def api_webhooks():
    """Webhook yönetimi"""
    if request.method == "GET":
        targets = webhook_notifier.get_targets()
        stats = webhook_notifier.get_stats()
        return jsonify({"targets": targets, "stats": stats})
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        name = data.get("name", "")
        url = data.get("url", "")
        if not name or not url:
            return jsonify({"status": "error", "message": "name ve url gerekli"}), 400
        try:
            webhook_notifier.add_target(name, url)
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        return jsonify({"status": "ok", "targets": webhook_notifier.get_targets()})

    data = request.get_json(silent=True) or {}
    name = data.get("name", "")
    webhook_notifier.remove_target(name)
    return jsonify({"status": "ok", "targets": webhook_notifier.get_targets()})


# -- WebSocket --
@socketio.on("connect")
def on_connect():
    emit(
        "config",
        {
            "confidence": conf_thresh,
            "clahe_clip": clahe_clip,
            "recording": is_recording,
            "overlay_clahe": overlay_clahe,
            "overlay_boxes": overlay_boxes,
            "token_required": bool(config.API_TOKEN),
        },
    )


@socketio.on("get_stats")
def on_stats():
    stats = get_stats()
    stats["fps"] = buffer.fps
    stats["detections"] = buffer.count
    stats["gps"] = gps_state.get_dict()
    emit("stats", stats)


# -- Producer Thread: Sadece Kameradan Oku --
def camera_producer():
    """Surekli kameradan kare okuyarak guncel tutar (30 FPS)."""
    try:
        cam = Camera(
            width=config.CAM_WIDTH,
            height=config.CAM_HEIGHT,
            fps=config.TARGET_FPS,
        ).start()
    except Exception as exc:
        print(f"Kamera hatasi: {exc}")
        return

    prev_time = time.time()
    while True:
        try:
            frame = cam.get_frame()
            if frame is None:
                socketio.sleep(0.01)
                continue

            now = time.time()
            buffer.fps = 1.0 / (now - prev_time) if now > prev_time else 0
            prev_time = now

            buffer.update(raw=frame)

            try:
                if frame_queue.full():
                    frame_queue.get_nowait()
                frame_queue.put_nowait(frame.copy())
            except queue.Empty:
                pass

            socketio.sleep(1.0 / config.TARGET_FPS)
        except Exception as exc:
            print(f"Producer Hata: {exc}")
            socketio.sleep(0.1)


# -- Consumer Thread: Sadece Tespit Yap --
def detection_loop():
    global is_recording

    try:
        detector = Detector(config.MODEL_PATH)
    except Exception as exc:
        print(f"Model hatasi: {exc}")
        return

    ensure_directory(config.THUMB_DIR)
    ensure_csv_header(CSV_LOG_FILE)

    last_save_time = 0.0

    while True:
        try:
            frame = frame_queue.get()
            clahe_frame = apply_clahe(frame.copy(), clip=clahe_clip, grid=config.CLAHE_GRID)
            boxes, confs = detector.detect(frame, conf=conf_thresh, use_clahe=True, clahe_clip=clahe_clip)

            for box, conf in zip(boxes, confs):
                if conf < conf_thresh or not is_recording:
                    continue

                now_time = time.time()
                if now_time - last_save_time < config.DASHBOARD_SAVE_INTERVAL:
                    break

                now_dt = datetime.now()
                ts = now_dt.strftime("%H%M%S_%f")

                thumbnail_name = f"t_{ts}.jpg"
                thumbnail_path = save_thumbnail(
                    frame,
                    box,
                    config.THUMB_DIR,
                    thumbnail_name,
                    max_files=config.MAX_THUMB_FILES,
                )
                if thumbnail_path is not None:
                    socketio.emit(
                        "detection",
                        {
                            "timestamp": now_dt.strftime("%H:%M:%S"),
                            "confidence": round(float(conf), 2),
                            "thumbnail": thumbnail_path.name,
                        },
                    )

                rows = build_detection_rows([box], [conf], now_dt=now_dt)
                if rows:
                    csv_log_queue.put(rows[0])

                lat, lon, gps_ts, gps_valid = gps_state.get()
                if gps_valid:
                    try:
                        insert_detection("Pufferfish", conf, lat, lon, now_time)
                        socketio.emit(
                            "gis_detection",
                            {
                                "lat": lat,
                                "lon": lon,
                                "confidence": round(float(conf), 2),
                                "timestamp": now_dt.strftime("%H:%M:%S"),
                            },
                        )
                        socketio.sleep(0)
                    except Exception as spatial_err:
                        print(f"Spatial Log Hata: {spatial_err}")

                last_save_time = now_time
                webhook_notifier.notify_async(
                    species="Lagocephalus sceleratus",
                    confidence=round(float(conf), 4),
                    lat=lat if gps_valid else None,
                    lon=lon if gps_valid else None,
                    timestamp=now_dt.strftime("%Y-%m-%d %H:%M:%S"),
                )
                break

            dets = [(x1, y1, x2, y2, float(conf)) for (x1, y1, x2, y2), conf in zip(boxes, confs)]
            buffer.update(detections=dets, clahe=clahe_frame)
            socketio.sleep(0)
        except Exception as exc:
            print(f"Consumer Hata: {exc}")
            socketio.sleep(0.1)


# -- Stream Thread: Gorselleri Birlestir --
def render_loop():
    global overlay_boxes, overlay_clahe

    while True:
        try:
            raw = buffer.get("raw")
            clahe_frame = buffer.get("clahe")
            if raw is not None:
                base_frame = clahe_frame.copy() if overlay_clahe and clahe_frame is not None else raw.copy()
                if overlay_boxes:
                    det_frame = draw_boxes(
                        base_frame,
                        [d[:4] for d in buffer.detections],
                        [d[4] for d in buffer.detections],
                    )
                else:
                    det_frame = base_frame
                buffer.update(detection=det_frame)
            socketio.sleep(0.03)
        except Exception:
            socketio.sleep(0.1)


# -- WS Base64 Stream Loop --
def ws_stream_loop():
    """Base64 kodlanmış frameleri WebSocket üzerinden gönderir."""
    interval = 1.0 / config.STREAM_FPS
    while True:
        try:
            frame_b64 = get_base64_frame(buffer, stream_type="detection", quality=config.WS_STREAM_QUALITY)
            if frame_b64:
                socketio.emit("ws_frame", {"image": frame_b64, "type": "detection"})
            socketio.sleep(interval)
        except Exception:
            socketio.sleep(0.1)


# -- Stats emitter --
def stats_loop():
    while True:
        stats = get_stats()
        stats["fps"] = round(buffer.fps, 1)
        stats["detections"] = buffer.count
        stats["gps"] = gps_state.get_dict()
        confs = [d[4] for d in buffer.detections]
        stats["confidence"] = max(confs) if confs else 0.0
        socketio.emit("stats", stats)
        socketio.sleep(1)


# -- Main --
def main():
    print("=" * 40)
    print("Balon Baligi Dashboard")
    print("=" * 40)

    try:
        init_db()
    except Exception as exc:
        print(f"DB Init Error: {exc}")

    socketio.start_background_task(camera_producer)
    socketio.start_background_task(detection_loop)
    socketio.start_background_task(render_loop)
    socketio.start_background_task(ws_stream_loop)
    socketio.start_background_task(stats_loop)
    socketio.start_background_task(gps_reader_thread)
    socketio.start_background_task(csv_logger_thread)

    print(f"http://0.0.0.0:{config.DASHBOARD_PORT}")
    print("=" * 40)

    socketio.run(app, host="0.0.0.0", port=config.DASHBOARD_PORT, debug=False)


if __name__ == "__main__":
    main()
