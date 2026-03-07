import time
import numpy as np
from app.core.camera import CameraThread

def test_camera_fallback_initialization():
    # Will use fallback as PiCamera2 won't be available on PC
    cam = CameraThread(width=640, height=480, fps=10, use_pi=False)
    # Check if the fallback correctly initialized a blank frame
    assert cam.blank_frame.shape == (480, 640, 3)
    
    # Do a manual read
    frame = cam.read()
    assert frame is not None
    # If the camera couldn't be opened, it will return the blank error frame
    assert frame.shape == (480, 640, 3)

def test_camera_thread_start_stop():
    cam = CameraThread(use_pi=False)
    cam.start()
    
    # Thread should be running
    assert cam._running is True
    assert cam._thread.is_alive()
    
    # Try to grab a frame for up to 5 seconds
    frame = None
    for _ in range(50):
        frame = cam.get_frame()
        if frame is not None:
            break
        time.sleep(0.1)
        
    assert frame is not None
    
    cam.stop()
    assert cam._running is False
