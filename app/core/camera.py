# Kamera modulu - Pi ve PC destegi
import cv2

class Camera:
    def __init__(self, width=640, height=480, fps=30, use_pi=False):
        self.width = width
        self.height = height
        self.use_pi = use_pi
        self.cap = None
        self.picam = None
        
        if use_pi:
            self._init_picamera()
        else:
            self._init_opencv()
    
    def _init_picamera(self):
        try:
            from picamera2 import Picamera2
            self.picam = Picamera2()
            config = self.picam.create_video_configuration(
                main={"size": (self.width, self.height)},
                controls={"FrameDurationLimits": (33333, 33333)}
            )
            self.picam.configure(config)
            self.picam.start()
            print("Pi kamera baslatildi")
        except Exception as e:
            print(f"Pi kamera hatasi: {e}, OpenCV'ye geciliyor")
            self._init_opencv()
            self.use_pi = False
    
    def _init_opencv(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not self.cap.isOpened():
            raise RuntimeError("Kamera acilamadi")
        print("OpenCV kamera baslatildi")
    
    def read(self):
        if self.use_pi and self.picam:
            frame = self.picam.capture_array()
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = self.cap.read()
            return frame if ret else None
    
    def release(self):
        if self.cap:
            self.cap.release()
        if self.picam:
            self.picam.close()
