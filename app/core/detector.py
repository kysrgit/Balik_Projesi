# YOLO tespit modulu
from ultralytics import YOLO

class Detector:
    def __init__(self, model_path):
        self.model = YOLO(str(model_path), task='detect')
        print(f"Model yuklendi: {model_path}")
    
    def detect(self, frame, conf=0.6):
        """Tek frame uzerinde tespit yap, (boxes, confs) dondur"""
        results = self.model.predict(source=frame, conf=conf, imgsz=640, verbose=False)
        
        boxes = []
        confs = []
        
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            c = float(box.conf)
            boxes.append((x1, y1, x2, y2))
            confs.append(c)
        
        return boxes, confs
