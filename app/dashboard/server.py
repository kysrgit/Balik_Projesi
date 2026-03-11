#!/usr/bin/env python3
"""
Dashboard Web Server
Flask + SocketIO ile gercek zamanli izleme paneli
"""
import eventlet
eventlet.monkey_patch()

import os
import sys
import time
import queue
import json
import cv2
import csv
from datetime import datetime
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit

# Path ayari
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.core import config, Detector
from app.utils import draw_boxes
from app.dashboard.stream import FrameBuffer, generate_mjpeg, get_base64_frame
from app.core import Camera # Moved Camera import here as it's no longer from app.core directly
from app.core.gps import gps_state, gps_reader_thread
from app.db.spatial import init_db, insert_detection
from app.export import to_geojson, to_csv_download, to_darwincore_archive, WebhookNotifier

# Flask app
app = Flask(__name__)
# Guvenlik: Secret key ortam degiskeninden alinir
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', os.urandom(32).hex())
socketio = SocketIO(app, cors_allowed_origins=config.ALLOWED_ORIGINS, async_mode='eventlet')

# Global state initialized from config
buffer = FrameBuffer()
conf_thresh = config.CONF_THRESH
clahe_clip = config.CLAHE_CLIP
is_recording = False
log = []

# Frame Queue for decoupled inference
frame_queue = queue.Queue(maxsize=5)

# Init CSV logger
CSV_LOG_FILE = "detections_log.csv"
csv_log_queue = queue.Queue()

def csv_logger_thread():
    # Write header if file doesn't exist
    if not os.path.exists(CSV_LOG_FILE):
        with open(CSV_LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Date", "Time", "Confidence", "BBox_X1", "BBox_Y1", "BBox_X2", "BBox_Y2"])

    with open(CSV_LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        while True:
            row = csv_log_queue.get()
            writer.writerow(row)
            f.flush()

if not os.path.exists(CSV_LOG_FILE):
    with open(CSV_LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Date", "Time", "Confidence", "BBox_X1", "BBox_Y1", "BBox_X2", "BBox_Y2"])


# -- Sistem bilgileri --
def get_stats():
    stats = {'cpu_temp': 0, 'throttled': False, 'fan_rpm': 0}
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            stats['cpu_temp'] = int(f.read().strip()) / 1000
    except:
        pass
    return stats


# -- Routes --
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video/<stream_type>')
def video(stream_type):
    return Response(generate_mjpeg(buffer, stream_type),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    global conf_thresh, clahe_clip, is_recording
    if request.method == 'POST':
        data = request.json
        if 'confidence' in data:
            conf_thresh = float(data['confidence'])
        if 'clahe_clip' in data:
            clahe_clip = float(data['clahe_clip'])
        return jsonify({'status': 'ok'})
    return jsonify({'confidence': conf_thresh, 'clahe_clip': clahe_clip, 'recording': is_recording})

@app.route('/api/snapshot', methods=['POST'])
def snapshot():
    frame = buffer.get('detection')
    if frame is not None:
        os.makedirs('detections', exist_ok=True)
        name = f"snap_{datetime.now().strftime('%H%M%S')}.jpg"
        cv2.imwrite(f"detections/{name}", frame)
        return jsonify({'status': 'ok', 'file': name})
    return jsonify({'status': 'error'})

@app.route('/api/record', methods=['POST'])
def record_toggle():
    global is_recording
    is_recording = not is_recording
    return jsonify({'status': 'ok', 'recording': is_recording})

@app.route('/detections/<path:name>')
def serve_file(name):
    return send_from_directory('detections', name)


# -- Export / Data Sharing Endpoints --
webhook_notifier = WebhookNotifier(rate_limit_seconds=60)

@app.route('/api/export/geojson')
def export_geojson():
    """GeoJSON export — harita servisleri, QGIS, Leaflet uyumlu"""
    data = to_geojson(CSV_LOG_FILE, str(config.DB_PATH))
    return Response(
        json.dumps(data, ensure_ascii=False, indent=2),
        mimetype='application/geo+json',
        headers={'Content-Disposition': 'attachment; filename=pufferfish_detections.geojson'}
    )

@app.route('/api/export/csv')
def export_csv():
    """CSV download — araştırmacılar için"""
    csv_data = to_csv_download(CSV_LOG_FILE, str(config.DB_PATH))
    return Response(
        csv_data,
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=pufferfish_detections.csv'}
    )

@app.route('/api/export/darwincore')
def export_darwincore():
    """DarwinCore Archive (ZIP) — GBIF / OBIS uyumlu"""
    zip_data = to_darwincore_archive(CSV_LOG_FILE, str(config.DB_PATH))
    return Response(
        zip_data,
        mimetype='application/zip',
        headers={'Content-Disposition': 'attachment; filename=pufferfish_dwca.zip'}
    )

@app.route('/api/webhooks', methods=['GET', 'POST', 'DELETE'])
def api_webhooks():
    """Webhook yönetimi"""
    if request.method == 'GET':
        targets = webhook_notifier.get_targets()
        stats = webhook_notifier.get_stats()
        return jsonify({'targets': targets, 'stats': stats})
    elif request.method == 'POST':
        data = request.json or {}
        name = data.get('name', '')
        url = data.get('url', '')
        if not name or not url:
            return jsonify({'status': 'error', 'message': 'name ve url gerekli'}), 400
        webhook_notifier.add_target(name, url)
        return jsonify({'status': 'ok', 'targets': webhook_notifier.get_targets()})
    elif request.method == 'DELETE':
        data = request.json or {}
        name = data.get('name', '')
        webhook_notifier.remove_target(name)
        return jsonify({'status': 'ok', 'targets': webhook_notifier.get_targets()})


# -- WebSocket --
@socketio.on('connect')
def on_connect():
    emit('config', {'confidence': conf_thresh, 'clahe_clip': clahe_clip, 'recording': is_recording})

@socketio.on('get_stats')
def on_stats():
    stats = get_stats()
    stats['fps'] = buffer.fps
    stats['detections'] = buffer.count
    stats['gps'] = gps_state.get_dict()
    emit('stats', stats)


# -- Producer Thread: Sadece Kameradan Oku --
def camera_producer():
    """Surekli kameradan kare okuyarak guncel tutar (30 FPS)"""

    # Yeni native pi5 libcamera uzerinden baslar, thread asenkron çalışır
    try:
        cam = Camera(width=config.CAM_WIDTH, height=config.CAM_HEIGHT, fps=config.TARGET_FPS).start()
    except Exception as e:
        print(f"Kamera hatasi: {e}")
        return

    prev_time = time.time()
    while True:
        try:
            frame = cam.read()
            if frame is None:
                socketio.sleep(0.01)
                continue

            now = time.time()
            # Yalnızca kare değiştiğinde FPS hesabını güncellemek için kaba bir koruma
            # Ancak wait vs kullanılmadığı için her döngüde frame alınabilir, bu yüzden uyuma ekliyoruz.
            buffer.fps = 1.0 / (now - prev_time) if now > prev_time else 0
            prev_time = now

            # Display buffer update
            buffer.update(raw=frame)

            # Send latest raw frame to inference queue
            try:
                # Drop oldest frame if queue full to keep real-time
                if frame_queue.full():
                    frame_queue.get_nowait()
                frame_queue.put_nowait(frame.copy())
            except:
                pass

            # CPU döngü koruması, 30fps ~= 33ms
            socketio.sleep(1.0 / config.TARGET_FPS)

        except Exception as e:
            print(f"Producer Hata: {e}")
            socketio.sleep(0.1)


# -- Consumer Thread: Sadece Tespit Yap --
def detection_loop():
    global is_recording

    # Model yukle
    try:
        detector = Detector(config.MODEL_PATH)
    except Exception as e:
        print(f"Model hatasi: {e}")
        return

    os.makedirs('detections/thumbs', exist_ok=True)

    last_save_time = 0.0

    while True:
        try:
            # Wait for next frame
            frame = frame_queue.get()

            # Tespit (CLAHE sadece inference edilen frame'e ve kucultulmus tensore uygulanacak)
            boxes, confs = detector.detect(frame, conf=conf_thresh, use_clahe=True, clahe_clip=clahe_clip)

            # Olay bazli islemler
            for (x1, y1, x2, y2), c in zip(boxes, confs):
                if c >= conf_thresh and is_recording: # Kayit acikken, dashboard slider esigini kullan
                    now_time = time.time()

                    # Rate limiting: Ziplamalari ve disk yorgunlugunu engelle
                    if now_time - last_save_time >= config.DASHBOARD_SAVE_INTERVAL:
                        now_dt = datetime.now()
                        ts = now_dt.strftime('%H%M%S_%f')

                        # 1. Bildirim Icin Thumbnail
                        thumb = frame[max(0,y1-10):y2+10, max(0,x1-10):x2+10]
                        if thumb.size > 0:
                            thumbnail_name = f"t_{ts}.jpg"
                            path = f"detections/thumbs/{thumbnail_name}"
                            cv2.imwrite(path, cv2.resize(thumb, (100, 100)))
                            socketio.emit('detection', {
                                'timestamp': now_dt.strftime('%H:%M:%S'),
                                'confidence': round(c, 2),
                                'thumbnail': thumbnail_name
                            })

                        # 2. Kalici Veri Sistikcasi Icin CSV Kayit
                        csv_log_queue.put([
                            ts, now_dt.strftime('%Y-%m-%d'), now_dt.strftime('%H:%M:%S'),
                            round(c, 4), x1, y1, x2, y2
                        ])

                        # 3. Canli GIS Loglama: Eger GPS verisi gecerliyse
                        lat, lon, gps_ts, is_valid = gps_state.get()
                        if is_valid:
                            try:
                                # SpatiaLite Log
                                insert_detection("Pufferfish", c, lat, lon, now_time)

                                # Frontend'e event yolla
                                socketio.emit('gis_detection', {
                                    'lat': lat,
                                    'lon': lon,
                                    'confidence': round(c, 2),
                                    'timestamp': now_dt.strftime('%H:%M:%S')
                                })
                                # Kritik: CPU context switch izin vermesi icin eventlet sleep
                                socketio.sleep(0)
                            except Exception as spatial_err:
                                print(f"Spatial Log Hata: {spatial_err}")

                        last_save_time = now_time

                        # 4. Webhook bildirimi (arka planda, ana thread'i bloklamaz)
                        webhook_notifier.notify_async(
                            species="Lagocephalus sceleratus",
                            confidence=round(c, 4),
                            lat=lat if is_valid else None,
                            lon=lon if is_valid else None,
                            timestamp=now_dt.strftime('%Y-%m-%d %H:%M:%S')
                        )

                        break # Bu frame icin ilk gecerli objeyi (en yuksek guven) loglamak yeterlidir

            # Buffer'i guncelle (Diger asenkron thread cizecek)
            dets = [(x1, y1, x2, y2, c) for (x1, y1, x2, y2), c in zip(boxes, confs)]
            buffer.update(detections=dets)

            # CPU serbest bırakma, cooperations sağlar
            socketio.sleep(0)

        except Exception as e:
            print(f"Consumer Hata: {e}")
            socketio.sleep(0.1)


# -- Stream Thread: Gorselleri Birlestir --
def render_loop():
    while True:
        try:
            raw = buffer.get('raw')
            if raw is not None:
                # Kutulari bellekteki guncel durumdan ciz
                det_frame = draw_boxes(raw.copy(), [d[:4] for d in buffer.detections], [d[4] for d in buffer.detections])
                buffer.update(detection=det_frame)
            socketio.sleep(0.03) # 30fps render limit
        except:
             socketio.sleep(0.1)

# -- WS Base64 Stream Loop --
def ws_stream_loop():
    """Base64 kodlanmış frameleri WebSocket üzerinden gönderir (Alternatif Stream)"""
    interval = 1.0 / config.STREAM_FPS
    while True:
        try:
            # Şimdilik sadece detection stream'ini gönderiyoruz
            frame_b64 = get_base64_frame(buffer, stream_type='detection', quality=config.WS_STREAM_QUALITY)
            if frame_b64:
                socketio.emit('ws_frame', {'image': frame_b64, 'type': 'detection'})
            socketio.sleep(interval)
        except Exception as e:
            socketio.sleep(0.1)

# -- Stats emitter --
def stats_loop():
    while True:
        stats = get_stats()
        stats['fps'] = round(buffer.fps, 1)
        stats['detections'] = buffer.count
        stats['gps'] = gps_state.get_dict()
        confs = [d[4] for d in buffer.detections]
        stats['confidence'] = max(confs) if confs else 0.0
        socketio.emit('stats', stats)
        socketio.sleep(1)


# -- Main --
def main():
    print("=" * 40)
    print("Balon Baligi Dashboard")
    print("=" * 40)

    # DB Init
    try:
        init_db()
    except Exception as e:
        print(f"SpatiaLite DB Init Error: {e}")

    # Eventlet thread'leri (GreenThread) başlatılır
    socketio.start_background_task(camera_producer)
    socketio.start_background_task(detection_loop)
    socketio.start_background_task(render_loop)
    socketio.start_background_task(ws_stream_loop)
    socketio.start_background_task(stats_loop)

    # GPS okumaları Eventlet sleep desteklesin diye monkey patch ile tam uyumludur
    socketio.start_background_task(gps_reader_thread)

    # Arka plan CSV loglama thread'i
    socketio.start_background_task(csv_logger_thread)

    print(f"http://0.0.0.0:{config.DASHBOARD_PORT}")
    print("=" * 40)

    socketio.run(app, host='0.0.0.0', port=config.DASHBOARD_PORT,
                 debug=False, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    main()
