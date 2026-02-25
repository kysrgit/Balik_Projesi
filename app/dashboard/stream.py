# MJPEG streaming ve frame buffer
import threading
import time
import cv2

class FrameBuffer:
    """Thread-safe frame storage"""
    def __init__(self):
        self.raw = None
        self.clahe = None
        self.detection = None
        self.lock = threading.Lock()
        self.fps = 0
        self.detections = []
        self.last_conf = 0
        self.count = 0
    
    def update(self, raw=None, clahe=None, detection=None, detections=None):
        with self.lock:
            if raw is not None:
                self.raw = raw.copy()
            if clahe is not None:
                self.clahe = clahe.copy()
            if detection is not None:
                self.detection = detection.copy()
            if detections is not None:
                self.detections = detections
                self.count = len(detections)
                if detections:
                    self.last_conf = max(d[4] for d in detections)
    
    def get(self, stream_type):
        with self.lock:
            if stream_type == 'raw':
                return self.raw
            elif stream_type == 'clahe':
                return self.clahe
            else:
                return self.detection


def generate_mjpeg(buffer, stream_type='detection', target_fps=15):
    """MJPEG stream generator"""
    interval = 1.0 / target_fps
    last_time = 0
    
    while True:
        now = time.time()
        if now - last_time < interval:
            time.sleep(0.01)
            continue
        
        frame = buffer.get(stream_type)
        if frame is None:
            time.sleep(0.05)
            continue
        
        # Kucuk streamler icin resize
        if stream_type in ['raw', 'clahe']:
            frame = cv2.resize(frame, (320, 240))
        
        quality = 60 if stream_type != 'detection' else 70
        _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')
        
        last_time = now
