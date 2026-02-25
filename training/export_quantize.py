# ONNX export ve INT8 quantization
import os
import glob
import numpy as np
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat

MODEL_PT = Path(__file__).parent.parent / "models" / "yolo11m_pufferfish.pt"
MODEL_ONNX = MODEL_PT.with_suffix('.onnx')
MODEL_INT8 = Path(__file__).parent.parent / "models" / "pufferfish_pi_int8.onnx"
CALIB_DIR = Path(__file__).parent.parent / "dataset" / "images" / "val"


class CalibReader(CalibrationDataReader):
    """Kalibrasyon icin gorsel okuyucu"""
    def __init__(self, img_dir):
        paths = glob.glob(str(img_dir / "*.jpg")) + glob.glob(str(img_dir / "*.png"))
        self.paths = paths[:100]  # max 100 gorsel
        self.idx = 0
        print(f"Kalibrasyon: {len(self.paths)} gorsel")
    
    def get_next(self):
        if self.idx >= len(self.paths):
            return None
        
        img = Image.open(self.paths[self.idx]).convert("RGB").resize((640, 640))
        data = np.array(img).astype(np.float32) / 255.0
        data = np.expand_dims(data.transpose(2, 0, 1), 0)
        
        self.idx += 1
        return {"images": data}


def export():
    if not MODEL_PT.exists():
        print(f"Model yok: {MODEL_PT}")
        return
    
    # ONNX export
    print("ONNX'e cevriliyor...")
    model = YOLO(MODEL_PT)
    model.export(format='onnx', opset=12, simplify=True)
    
    if not MODEL_ONNX.exists():
        print("Export basarisiz")
        return
    
    # INT8 quantization
    print("INT8 quantization...")
    reader = CalibReader(CALIB_DIR)
    
    quantize_static(
        model_input=str(MODEL_ONNX),
        model_output=str(MODEL_INT8),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8
    )
    
    print(f"Tamamlandi: {MODEL_INT8}")


if __name__ == "__main__":
    export()
