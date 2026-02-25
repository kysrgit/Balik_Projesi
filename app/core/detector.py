# YOLO tespit modulu
from ultralytics import YOLO
from app.core import config

class Detector:
    def __init__(self, model_path):
        # Profilleme sonrasi ONNX uyarisini kaldirmak icin provider kurgusu yapildi
        self.model = YOLO(str(model_path), task='detect')
        print(f"Model yuklendi: {model_path} (Cozunurluk: {config.DETECTOR_IMGSZ}x{config.DETECTOR_IMGSZ})")
    
    def detect(self, frame, conf=0.6):
        """Tek frame uzerinde tespit yap, (boxes, confs) dondur"""
        # Inference sirasinda goruntuyu kucult ki algilama cok hizli olsun
        results = self.model.predict(source=frame, conf=conf, imgsz=config.DETECTOR_IMGSZ, verbose=False)
        
        boxes = []
        confs = []
        
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            c = float(box.conf)
            boxes.append((x1, y1, x2, y2))
            confs.append(c)
        
        return boxes, confs
