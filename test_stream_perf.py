import time
import cv2
import numpy as np
import base64
from app.dashboard.stream import FrameBuffer, get_base64_frame

def benchmark():
    buffer = FrameBuffer()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    buffer.update(raw=frame, clahe=frame, detection=frame)

    start_time = time.time()
    for _ in range(100):
        get_base64_frame(buffer, stream_type='raw')
    end_time = time.time()

    print(f"Time for 100 calls (raw): {end_time - start_time:.4f} seconds")

    start_time = time.time()
    for _ in range(100):
        get_base64_frame(buffer, stream_type='detection')
    end_time = time.time()

    print(f"Time for 100 calls (detection): {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    benchmark()
