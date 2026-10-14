# YOLO tespit modulu
from ultralytics import YOLO
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
        
        # XNNPACKExecutionProvider'i yalnizca bu onnxruntime derlemesinde varsa ekle
        # (PyPI'daki aarch64 paketinde yok; olmayan saglayiciyi istemek EP hatasi basar)
        if providers is None:
            providers = ['CPUExecutionProvider']
        if ('XNNPACKExecutionProvider' in ort.get_available_providers()
                and 'XNNPACKExecutionProvider' not in providers):
            # En yuksek oncelikli olarak eklendi
            providers = ['XNNPACKExecutionProvider', *providers]
            
        super().__init__(path_or_bytes, sess_options=sess_options, providers=providers, provider_options=provider_options, **kwargs)

ort.InferenceSession = PatchedInferenceSession
# --------------------------------------------------------------------------

class Detector:
    def __init__(self, model_path):
        # Profilleme sonrasi ONNX uyarisini kaldirmak icin provider kurgusu yapildi
        self.model = YOLO(str(model_path), task='detect')
        print(f"Model yuklendi: {model_path} (Cozunurluk: {config.DETECTOR_IMGSZ}x{config.DETECTOR_IMGSZ})")
    
    def detect(self, frame, conf=0.6, use_clahe=False, clahe_clip=3.0):
        """Tek frame uzerinde tespit yap, (boxes, confs) dondur.

        Kare oldugu gibi verilir; ultralytics en-boy oranini koruyarak 640'a
        letterbox yapar (egitimdeki on isleme). Eskiden kare 640x640'a geriliyordu,
        bu 4:3 kamera karelerinde baliklari dikey uzatip egitimle uyumsuzluk yaratiyordu.
        Kutular dogrudan orijinal karenin koordinatlarinda doner.
        """
        img = apply_clahe(frame, clip=clahe_clip) if use_clahe else frame
        results = self.model.predict(source=img, conf=conf, imgsz=config.DETECTOR_IMGSZ, verbose=False)

        boxes_obj = results[0].boxes
        if boxes_obj is None or len(boxes_obj) == 0:
            return [], []
        # CPU/NumPy aktarimini dongu disinda tek seferde yap
        xyxy_arr = boxes_obj.xyxy.cpu().numpy().astype(int)
        conf_arr = boxes_obj.conf.cpu().numpy()
        boxes = [tuple(map(int, b)) for b in xyxy_arr]
        confs = [float(c) for c in conf_arr]
        return boxes, confs
