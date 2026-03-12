import time
import cv2
import numpy as np
import base64

class FrameBufferCached2:
    def __init__(self):
        self.raw = None
        self.clahe = None
        self.detection = None
        # Use an integer frame_id that increments on update
        self.frame_id = {'raw': 0, 'clahe': 0, 'detection': 0}

    def update(self, raw=None, clahe=None, detection=None):
        if raw is not None:
            self.raw = raw.copy()
            self.frame_id['raw'] += 1
        if clahe is not None:
            self.clahe = clahe.copy()
            self.frame_id['clahe'] += 1
        if detection is not None:
            self.detection = detection.copy()
            self.frame_id['detection'] += 1

    def get(self, stream_type):
        if stream_type == 'raw':
            return self.raw
        elif stream_type == 'clahe':
            return self.clahe
        else:
            return self.detection

# Global/static cache to avoid changing FrameBuffer class
_b64_cache = {}

def get_base64_frame_cached2(buffer, stream_type='detection', quality=50):
    global _b64_cache

    frame = buffer.get(stream_type)
    if frame is None:
        return None

    # We can use the object identity (id(frame)) as cache key!
    # Because FrameBuffer.update makes a .copy(), a new frame will have a new id.
    # We need to clean up old entries to prevent memory leaks, so we only keep the latest.
    frame_id = id(frame)
    cache_key = (stream_type, quality)

    if cache_key in _b64_cache:
        cached_id, cached_b64 = _b64_cache[cache_key]
        if cached_id == frame_id:
            return cached_b64

    if stream_type in ['raw', 'clahe']:
        frame = cv2.resize(frame, (320, 240))

    _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    result = base64.b64encode(jpg).decode('utf-8')

    _b64_cache[cache_key] = (frame_id, result)
    return result

def benchmark():
    buffer = FrameBufferCached2()
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    buffer.update(raw=frame, clahe=frame, detection=frame)

    start_time = time.time()
    for _ in range(100):
        get_base64_frame_cached2(buffer, stream_type='raw')
    end_time = time.time()
    print(f"Time for 100 calls cached object ID (raw): {end_time - start_time:.4f} seconds")

    start_time = time.time()
    for _ in range(100):
        get_base64_frame_cached2(buffer, stream_type='detection')
    end_time = time.time()
    print(f"Time for 100 calls cached object ID (detection): {end_time - start_time:.4f} seconds")

    # Test update
    print("Updating buffer with new frame...")
    frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    buffer.update(raw=frame2, clahe=frame2, detection=frame2)

    start_time = time.time()
    get_base64_frame_cached2(buffer, stream_type='raw')
    end_time = time.time()
    print(f"Time for 1 call after update (raw): {end_time - start_time:.4f} seconds")

    start_time = time.time()
    get_base64_frame_cached2(buffer, stream_type='raw')
    end_time = time.time()
    print(f"Time for 1 call after update (raw, cached): {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    benchmark()
