#!/usr/bin/env python3
"""
Dashboard Web Server
Flask + SocketIO ile gercek zamanli izleme paneli
"""
import os
import sys
import time
import threading
import queue
import cv2
import csv
from datetime import datetime
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit

# Path ayari
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.core import config, Detector
from app.utils import apply_clahe, draw_boxes
from app.dashboard.stream import FrameBuffer, generate_mjpeg
from app.core import Camera # Moved Camera import here as it's no longer from app.core directly

# Flask app
app = Flask(__name__)
# 🛡️ Sentinel: Removed hardcoded secret key (Critical)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', os.urandom(32).hex())
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state
buffer = FrameBuffer()
conf_thresh = 0.60
clahe_clip = 3.0
log = []

# Frame Queue for decoupled inference
frame_queue = queue.Queue(maxsize=5)

# Init CSV logger
CSV_LOG_FILE = "detections_log.csv"
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
    global conf_thresh, clahe_clip
    if request.method == 'POST':
        data = request.json
        if 'confidence' in data:
            conf_thresh = float(data['confidence'])
        if 'clahe_clip' in data:
            clahe_clip = float(data['clahe_clip'])
        return jsonify({'status': 'ok'})
    return jsonify({'confidence': conf_thresh, 'clahe_clip': clahe_clip})

@app.route('/api/snapshot', methods=['POST'])
def snapshot():
    frame = buffer.get('detection')
    if frame is not None:
        os.makedirs('detections', exist_ok=True)
        name = f"snap_{datetime.now().strftime('%H%M%S')}.jpg"
        cv2.imwrite(f"detections/{name}", frame)
        return jsonify({'status': 'ok', 'file': name})
    return jsonify({'status': 'error'})

@app.route('/detections/<path:name>')
def serve_file(name):
    return send_from_directory('detections', name)


# -- WebSocket --
@socketio.on('connect')
def on_connect():
    emit('config', {'confidence': conf_thresh, 'clahe_clip': clahe_clip})

@socketio.on('get_stats')
def on_stats():
    stats = get_stats()
    stats['fps'] = buffer.fps
    stats['detections'] = buffer.count
    emit('stats', stats)


# -- Producer Thread: Sadece Kameradan Oku --
def camera_producer():
    """Surekli kameradan kare okuyarak guncel tutar (30 FPS)"""
    global clahe_clip
    
    # Yeni native pi5 libcamera uzerinden baslar
    try:
        cam = Camera(width=640, height=480, fps=30)
    except Exception as e:
        print(f"Kamera hatasi: {e}")
        return
        
    prev_time = time.time()
    while True:
        try:
            frame = cam.read()
            if frame is None:
                continue
                
            now = time.time()
            buffer.fps = 1.0 / (now - prev_time) if now > prev_time else 0
            prev_time = now
            
            # Display buffer update
            buffer.update(raw=frame.copy())
            
            # CLAHE
            clahe = apply_clahe(frame, clahe_clip)
            buffer.update(clahe=clahe)
            
            # Send latest clahe frame to inference queue
            try:
                # Drop oldest frame if queue full to keep real-time
                if frame_queue.full():
                    frame_queue.get_nowait()
                frame_queue.put_nowait(clahe.copy())
            except:
                pass

        except Exception as e:
            print(f"Producer Hata: {e}")
            time.sleep(0.1)


# -- Consumer Thread: Sadece Tespit Yap --
def detection_loop():
    global conf_thresh
    
    # Model yukle
    try:
        detector = Detector(config.MODEL_PATH)
    except Exception as e:
        print(f"Model hatasi: {e}")
        return
    
    os.makedirs('detections/thumbs', exist_ok=True)
    
    last_boxes = []
    last_confs = []
    
    while True:
        try:
            # Wait for next frame
            frame = frame_queue.get()
            
            # Tespit
            boxes, confs = detector.detect(frame, conf_thresh)
            last_boxes, last_confs = boxes, confs
            
            # Olay bazli islemler
            for (x1, y1, x2, y2), c in zip(boxes, confs):
                if c > 0.70: # Test bittigi icin uyarilari gercek degere (0.70) aldik
                    now = datetime.now()
                    ts = now.strftime('%H%M%S_%f')
                    
                    # 1. Bildirim Icin Thumbnail
                    thumb = frame[max(0,y1-10):y2+10, max(0,x1-10):x2+10]
                    if thumb.size > 0:
                        thumbnail_name = f"t_{ts}.jpg"
                        path = f"detections/thumbs/{thumbnail_name}"
                        cv2.imwrite(path, cv2.resize(thumb, (100, 100)))
                        socketio.emit('detection', {
                            'timestamp': now.strftime('%H:%M:%S'),
                            'confidence': round(c, 2),
                            'thumbnail': thumbnail_name
                        })
                    
                    # 2. Kalici Veri Sistikcasi Icin CSV Kayit
                    with open(CSV_LOG_FILE, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            ts, now.strftime('%Y-%m-%d'), now.strftime('%H:%M:%S'),
                            round(c, 4), x1, y1, x2, y2
                        ])
            
            # Buffer'i guncelle (Diger asenkron thread cizecek)
            dets = [(x1, y1, x2, y2, c) for (x1, y1, x2, y2), c in zip(boxes, confs)]
            buffer.update(detections=dets)
            
        except Exception as e:
            print(f"Consumer Hata: {e}")
            time.sleep(0.1)


# -- Stream Thread: Gorselleri Birlestir --
def render_loop():
    while True:
        try:
            clahe = buffer.get('clahe')
            if clahe is not None:
                # Kutulari bellekteki guncel durumdan ciz
                det_frame = draw_boxes(clahe.copy(), [d[:4] for d in buffer.detections], [d[4] for d in buffer.detections])
                buffer.update(detection=det_frame)
            time.sleep(0.03) # 30fps render limit
        except:
             time.sleep(0.1)


# -- Stats emitter --
def stats_loop():
    while True:
        stats = get_stats()
        stats['fps'] = round(buffer.fps, 1)
        stats['detections'] = buffer.count
        confs = [d[4] for d in buffer.detections]
        stats['confidence'] = max(confs) if confs else 0.0
        socketio.emit('stats', stats)
        time.sleep(1)


# -- Main --
def main():
    print("=" * 40)
    print("Balon Baligi Dashboard")
    print("=" * 40)
    
    # Thread'leri baslat (Asenkron ucloud yapi)
    threading.Thread(target=camera_producer, daemon=True).start()
    threading.Thread(target=detection_loop, daemon=True).start()
    threading.Thread(target=render_loop, daemon=True).start()
    threading.Thread(target=stats_loop, daemon=True).start()
    
    print(f"http://0.0.0.0:{config.DASHBOARD_PORT}")
    print("=" * 40)
    
    socketio.run(app, host='0.0.0.0', port=config.DASHBOARD_PORT, 
                 debug=False, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    main()
