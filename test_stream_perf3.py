import time
import cv2
import numpy as np
import base64
from app.dashboard.stream import FrameBuffer

class FrameBufferCached(FrameBuffer):
    def __init__(self):
        super().__init__()
        # Cache for generate_mjpeg
        self.mjpeg_cache = {}
        # Cache for get_base64_frame
        self.b64_cache = {}

    def update(self, raw=None, clahe=None, detection=None, detections=None):
        with self.lock:
            # Check what changed to invalidate cache
            if raw is not None:
                self.raw = raw.copy()
                self.mjpeg_cache.pop('raw', None)
                self.b64_cache.pop('raw', None)
            if clahe is not None:
                self.clahe = clahe.copy()
                self.mjpeg_cache.pop('clahe', None)
                self.b64_cache.pop('clahe', None)
            if detection is not None:
                self.detection = detection.copy()
                self.mjpeg_cache.pop('detection', None)
                self.b64_cache.pop('detection', None)
            if detections is not None:
                self.detections = detections
                self.count = len(detections)
                if detections:
                    self.last_conf = max(d[4] for d in detections)

def get_base64_frame_cached(buffer, stream_type='detection', quality=50):
    """WebSocket üzerinden gönderim için frame'i base64 string yapar"""
    # Try to get from cache first
    cache_key = (stream_type, quality)

    with buffer.lock:
        if hasattr(buffer, 'b64_cache') and cache_key in buffer.b64_cache:
            return buffer.b64_cache[cache_key]

        frame = buffer.raw if stream_type == 'raw' else (buffer.clahe if stream_type == 'clahe' else buffer.detection)
        if frame is None:
            return None

        if stream_type in ['raw', 'clahe']:
            # we need a copy if we resize? Actually cv2.resize returns a new array
            frame = cv2.resize(frame, (320, 240))

        _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        result = base64.b64encode(jpg).decode('utf-8')

        if hasattr(buffer, 'b64_cache'):
            buffer.b64_cache[cache_key] = result

        return result

def benchmark():
    buffer = FrameBufferCached()
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    buffer.update(raw=frame, clahe=frame, detection=frame)

    start_time = time.time()
    for _ in range(100):
        get_base64_frame_cached(buffer, stream_type='raw')
    end_time = time.time()

    print(f"Time for 100 calls cached (raw): {end_time - start_time:.4f} seconds")

    start_time = time.time()
    for _ in range(100):
        get_base64_frame_cached(buffer, stream_type='detection')
    end_time = time.time()

    print(f"Time for 100 calls cached (detection): {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    benchmark()
