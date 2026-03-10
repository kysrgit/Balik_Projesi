# Kamera modulu - Pi 5 Native Libcamera + OpenCV Fallback Destegi
import cv2
import numpy as np
import time
import threading

class CameraThread:
    """Thread-safe asenkron kamera yakalama sinifi (RULE 1 uyumlu)"""
    def __init__(self, width=640, height=480, fps=30, use_pi=True):
        self.width = width
        self.height = height
        self.fps = fps
        self.picam = None
        self.cap = None  # OpenCV fallback
        self.error_msg = "Unknown Error"
        
        self._lock = threading.Lock()
        self._frame = None
        self._running = False
        self._stop_event = threading.Event()
        
        # Kamera hic acilmazsa basacagimiz siyah ekran
        self.blank_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        if use_pi:
            self._init_picamera()
        
        # Pi kamerasi acilmadiysa OpenCV ile dene (PC/USB webcam fallback)
        if self.picam is None:
            self._init_opencv()
            
    def _init_picamera(self):
        try:
            from picamera2 import Picamera2
            self.picam = Picamera2()
            
            # Formati otomatik birakip sadece cozunurluk zorluyoruz
            config = self.picam.create_video_configuration(main={"size": (self.width, self.height)})
            self.picam.configure(config)
            self.picam.start()
            print("=====================================")
            print("✅ Pi 5 Native Kamera (libcamera) basariyla baslatildi!")
            print("=====================================")
        except ImportError:
            self.error_msg = "picamera2 Kutuphanesi Kurulamadi! Sanal Ortam Hatasi."
            print(f"❌ {self.error_msg}")
            self.picam = None
        except Exception as e:
            self.error_msg = f"Fiziksel Kamera Hatasi: {str(e)[:50]}"
            print(f"❌ {self.error_msg}")
            self.picam = None

    def _init_opencv(self):
        """OpenCV VideoCapture fallback (PC/USB webcam)"""
        try:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                print("=====================================")
                print("✅ OpenCV Kamera (USB/Webcam) basariyla baslatildi!")
                print("=====================================")
            else:
                self.error_msg = "OpenCV: Kamera acilamadi (cihaz bulunamadi)"
                print(f"❌ {self.error_msg}")
                self.cap = None
        except Exception as e:
            self.error_msg = f"OpenCV Kamera Hatasi: {str(e)[:50]}"
            print(f"❌ {self.error_msg}")
            self.cap = None

    def start(self):
        """Kamera okuma döngüsünü arka planda başlatır."""
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return self
        
    def _capture_loop(self):
        """Asenkron frame toplama döngüsü."""
        while self._running:
            frame = self._read_raw()
            if frame is not None:
                with self._lock:
                    self._frame = frame
            else:
                # Olası bir donma durumunu engellemek için küçük bir bekleme
                self._stop_event.wait(0.01)

    def _read_raw(self):
        """Doğrudan cihazdan veya fallback'ten fiziksel frame okur."""
        # 1. Pi kamera
        if self.picam is not None:
            try:
                frame = self.picam.capture_array()
                if frame is not None:
                     # Picamera2 default olarak RGB verir, bunu OpenCV'nin istedigi BGR'ye ceviriyoruz
                     bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                     return bgr_frame
                return None
            except Exception as e:
                print(f"Kamera okuma hatasi: {e}")
                self._stop_event.wait(1.0)
                return None
        
        # 2. OpenCV fallback
        if self.cap is not None:
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    return frame
                return None
            except Exception as e:
                print(f"OpenCV okuma hatasi: {e}")
                self._stop_event.wait(1.0)
                return None
        
        # 3. Hic kamera yoksa hata frame'i goster
        error_frame = self.blank_frame.copy()
        cv2.putText(error_frame, "KAMERA HATASI!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.putText(error_frame, self.error_msg, (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(error_frame, "Lutfen Terminal ve Kablolari Kontrol Edin", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        self._stop_event.wait(0.5) # Bu ekran saniyede 2 kere guncellense yeter, CPU'yu yemeyelim
        return error_frame

    def get_frame(self):
        """Thread-safe son okunan kareyi döndür"""
        with self._lock:
            if self._frame is not None:
                return self._frame.copy()
            return None

    def read(self):
        """Geriye dönük uyumluluk ve basitlik için"""
        return self.get_frame()
    
    def stop(self):
        """Arka plan işlemini durdurur"""
        self._running = False
        self._stop_event.set()
        if hasattr(self, '_thread'):
            self._thread.join(timeout=1.0)
            
    def release(self):
        """Kaynakları temizler"""
        self.stop()
        if self.picam:
            self.picam.stop()
            self.picam.close()
        if self.cap:
            self.cap.release()

# İleri uyumluluk için alias
Camera = CameraThread
