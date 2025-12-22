#!/usr/bin/env python3
"""
Dashboard Web Server
Flask + SocketIO ile gercek zamanli izleme paneli
"""
import os
import sys
import time
import threading
import cv2
from datetime import datetime
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit

# Path ayari
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.core import config, Camera, Detector
from app.utils import apply_clahe, draw_boxes
from app.dashboard.stream import FrameBuffer, generate_mjpeg

# Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'balik2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state
buffer = FrameBuffer()
conf_thresh = 0.80
clahe_clip = 3.0
log = []


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


# -- Detection thread --
def detection_loop():
    global conf_thresh, clahe_clip
    
    # Model yukle
    try:
        detector = Detector(config.MODEL_PATH)
    except Exception as e:
        print(f"Model hatasi: {e}")
        return
    
    # Kamera
    try:
        is_pi = os.path.exists('/sys/class/thermal/thermal_zone0/temp')
        cam = Camera(config.CAM_WIDTH, config.CAM_HEIGHT, use_pi=is_pi)
    except Exception as e:
        print(f"Kamera hatasi: {e}")
        return
    
    os.makedirs('detections/thumbs', exist_ok=True)
    
    frame_n = 0
    prev_time = time.time()
    last_boxes = []
    last_confs = []
    
    while True:
        try:
            frame = cam.read()
            if frame is None:
                continue
            
            frame_n += 1
            buffer.update(raw=frame.copy())
            
            # CLAHE
            clahe = apply_clahe(frame, clahe_clip)
            buffer.update(clahe=clahe)
            
            # Her 5. frame'de tespit
            if frame_n % 5 == 0:
                boxes, confs = detector.detect(clahe, conf_thresh)
                last_boxes, last_confs = boxes, confs
                
                # Yuksek guvenli tespitleri logla
                for (x1, y1, x2, y2), c in zip(boxes, confs):
                    if c > 0.75:
                        ts = datetime.now().strftime('%H%M%S_%f')
                        thumb = frame[max(0,y1-10):y2+10, max(0,x1-10):x2+10]
                        if thumb.size > 0:
                            path = f"detections/thumbs/t_{ts}.jpg"
                            cv2.imwrite(path, cv2.resize(thumb, (100, 100)))
                            socketio.emit('detection', {
                                'time': datetime.now().strftime('%H:%M:%S'),
                                'conf': round(c, 2)
                            })
                
                # Buffer'a detections olarak kaydet
                dets = [(x1, y1, x2, y2, c) for (x1, y1, x2, y2), c in zip(boxes, confs)]
                buffer.update(detections=dets)
            
            # Kutulari ciz
            det_frame = draw_boxes(clahe, last_boxes, last_confs)
            buffer.update(detection=det_frame)
            
            # FPS
            now = time.time()
            buffer.fps = 1.0 / (now - prev_time)
            prev_time = now
            
            # Rate limit
            time.sleep(0.03)  # ~30fps
            
        except Exception as e:
            print(f"Hata: {e}")
            time.sleep(0.1)


# -- Stats emitter --
def stats_loop():
    while True:
        stats = get_stats()
        stats['fps'] = round(buffer.fps, 1)
        stats['detections'] = buffer.count
        socketio.emit('stats', stats)
        time.sleep(1)


# -- Main --
def main():
    print("=" * 40)
    print("Balon Baligi Dashboard")
    print("=" * 40)
    
    # Thread'leri baslat
    threading.Thread(target=detection_loop, daemon=True).start()
    threading.Thread(target=stats_loop, daemon=True).start()
    
    print(f"http://0.0.0.0:{config.DASHBOARD_PORT}")
    print("=" * 40)
    
    socketio.run(app, host='0.0.0.0', port=config.DASHBOARD_PORT, 
                 debug=False, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    main()
