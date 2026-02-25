# Kamera modulu - Pi 5 Native Libcamera Destegi
import cv2
import numpy as np
import time

class Camera:
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.picam = None
        self.error_msg = "Unknown Error"
        
        # Kamera hic acilmazsa basacagimiz siyah ekran
        self.blank_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        self._init_picamera()
    
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

    def read(self):
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
                time.sleep(1)
                return None
        else:
            # Eger Kamera modulu bozuksa veya hic yoksa, WEB EKRANINA Hatamizi Cizelim!
            error_frame = self.blank_frame.copy()
            cv2.putText(error_frame, "KAMERA HATASI!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(error_frame, self.error_msg, (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(error_frame, "Lutfen Terminal ve Kablolari Kontrol Edin", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            time.sleep(0.5) # Bu ekran saniyede 2 kere guncellense yeter, CPU'yu yemeyelim
            return error_frame
    
    def release(self):
        if self.picam:
            self.picam.stop()
            self.picam.close()
