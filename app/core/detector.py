# YOLO tespit modulu
from ultralytics import YOLO
import cv2
from app.core import config
from app.utils.image import apply_clahe
import onnxruntime as ort

# --- ONNX Runtime Optimizasyonlari (Pi 5 icin XNNPACK & Thread Tuning) ---
_original_session = ort.InferenceSession

class PatchedInferenceSession(_original_session):
    def __init__(self, path_or_bytes, sess_options=None, providers=None, provider_options=None, **kwargs):
        if sess_options is None:
            sess_options = ort.SessionOptions()
        
        # Pi 5 cekirdek sayisi (4)
        sess_options.intra_op_num_threads = 4
        # Spinning mekanizmasini devre disi birakarak gereksiz CPU %100 kullanimini ve isinmayi engelle (Throttle onleyici)
        sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        
        # XNNPACKExecutionProvider'i aktif et
        if providers is None:
            providers = ['XNNPACKExecutionProvider', 'CPUExecutionProvider']
        elif 'XNNPACKExecutionProvider' not in providers:
            # En yuksek oncelikli olarak eklendi
            providers.insert(0, 'XNNPACKExecutionProvider')
            
        super().__init__(path_or_bytes, sess_options=sess_options, providers=providers, provider_options=provider_options, **kwargs)

ort.InferenceSession = PatchedInferenceSession
# --------------------------------------------------------------------------

class Detector:
    def __init__(self, model_path):
        # Profilleme sonrasi ONNX uyarisini kaldirmak icin provider kurgusu yapildi
        self.model = YOLO(str(model_path), task='detect')
        print(f"Model yuklendi: {model_path} (Cozunurluk: {config.DETECTOR_IMGSZ}x{config.DETECTOR_IMGSZ})")
    
    def detect(self, frame, conf=0.6, use_clahe=True, clahe_clip=3.0):
        """Tek frame uzerinde tespit yap, (boxes, confs) dondur
           Performans optimizasyonu: Frame once kucultulur, sonra CLAHE uygulanir.
        """
        orig_h, orig_w = frame.shape[:2]
        tensor_size = (config.DETECTOR_IMGSZ, config.DETECTOR_IMGSZ)
        
        # Inference sirasinda goruntuyu kucult ki algilama cok hizli olsun
        # Ayrica algoritma sadeligi icin resize yapilip ustune CLAHE uygulaniyor
        img_resized = cv2.resize(frame, tensor_size)
        
        if use_clahe:
            img_resized = apply_clahe(img_resized, clip=clahe_clip)
            
        # Kucultulmus ve islenmis tensor ile predict
        results = self.model.predict(source=img_resized, conf=conf, imgsz=config.DETECTOR_IMGSZ, verbose=False)
        
        boxes = []
        confs = []
        
        # Kutulari orijinal cozunurluge geri olcekle
        x_scale = orig_w / tensor_size[0]
        y_scale = orig_h / tensor_size[1]
        
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0].cpu().numpy())
            
            x1 = int(x1 * x_scale)
            y1 = int(y1 * y_scale)
            x2 = int(x2 * x_scale)
            y2 = int(y2 * y_scale)
            
            c = float(box.conf)
            boxes.append((x1, y1, x2, y2))
            confs.append(c)
        
        return boxes, confs
