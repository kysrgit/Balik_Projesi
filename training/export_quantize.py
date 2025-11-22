import onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
import os
import numpy as np
from PIL import Image
from pathlib import Path
import glob
from ultralytics import YOLO

# --- CONFIGURATION ---
# Path to your trained model (adjust if your training run name changed)
MODEL_PATH = Path(__file__).parent.parent / "runs" / "detect" / "pufferfish_yolov8s" / "weights" / "best.pt"
ONNX_PATH = MODEL_PATH.with_suffix('.onnx')
QUANTIZED_MODEL_PATH = MODEL_PATH.parent / "best_int8.onnx"

# Calibration Data Path (uses your validation set)
CALIBRATION_IMG_DIR = Path(__file__).parent.parent / "dataset" / "images" / "val"

class YoloCalibrationDataReader(CalibrationDataReader):
    """
    Reads images from the validation set to calibrate the quantization.
    This ensures the INT8 model understands the range of values in real data.
    """
    def __init__(self, image_dir, input_name="images"):
        self.image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + \
                           glob.glob(os.path.join(image_dir, "*.png"))
        # Limit calibration to 100 images to save time
        self.image_paths = self.image_paths[:100]
        self.preprocess_flag = True
        self.enum_data_dicts = iter([])
        self.input_name = input_name
        print(f"[INFO] Calibration initialized with {len(self.image_paths)} images.")

    def get_next(self):
        iter_data = next(self.enum_data_dicts, None)
        if iter_data:
            return iter_data

        self.enum_data_dicts = None
        if self.preprocess_flag:
            self.preprocess_flag = False
            data_list = []
            for img_path in self.image_paths:
                # Preprocessing must match YOLOv8 runtime preprocessing
                # Resize to 640x640, Normalize 0-1
                img = Image.open(img_path).convert("RGB")
                img = img.resize((640, 640))
                img_data = np.array(img).astype(np.float32)
                img_data = img_data / 255.0  # Normalize 0-1
                img_data = img_data.transpose(2, 0, 1)  # HWC -> CHW
                img_data = np.expand_dims(img_data, axis=0) # Add batch dim
                
                data_list.append({self.input_name: img_data})
            
            self.enum_data_dicts = iter(data_list)
            return next(self.enum_data_dicts, None)
        return None

def export_and_quantize():
    # 1. Export PyTorch -> ONNX
    if not MODEL_PATH.exists():
        print(f"[ERROR] Trained model not found at {MODEL_PATH}")
        print("Please run train_yolo.py first!")
        return

    print(f"[INFO] Exporting {MODEL_PATH} to ONNX...")
    model = YOLO(MODEL_PATH)
    # Export arguments: opset=12 is stable, simplify=True removes redundant nodes
    model.export(format='onnx', opset=12, simplify=True)
    
    if not ONNX_PATH.exists():
        print("[ERROR] Export failed.")
        return

    # 2. Quantize ONNX -> INT8 (Static)
    print(f"[INFO] Starting INT8 Static Quantization...")
    
    # We need the input name of the model. Usually 'images' for YOLO.
    # We can inspect the model to be sure, but 'images' is standard for Ultralytics.
    dr = YoloCalibrationDataReader(CALIBRATION_IMG_DIR, input_name='images')

    quantize_static(
        model_input=str(ONNX_PATH),
        model_output=str(QUANTIZED_MODEL_PATH),
        calibration_data_reader=dr,
        quant_format=QuantFormat.QDQ, # Quantize-DeQuantize format (best for x86/ARM CPUs)
        per_channel=False,             # Per-tensor is usually faster on CPU
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8
    )
    
    print(f"[SUCCESS] Quantized model saved to: {QUANTIZED_MODEL_PATH}")
    print("Transfer this 'best_int8.onnx' file to your Raspberry Pi 5.")

if __name__ == "__main__":
    export_and_quantize()
