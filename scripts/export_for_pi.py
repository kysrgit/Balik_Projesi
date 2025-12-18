from ultralytics import YOLO
from onnxruntime.quantization import quantize_dynamic, QuantType
import os
import shutil

# Constants - paths relative to project root
MODEL_SOURCE = "models/yolo11m_pufferfish.pt"
FINAL_OUTPUT_NAME = "pufferfish_pi_int8.onnx"
TARGET_DIR = "models"

def main():
    print(f"--- Starting Manual INT8 Export for Raspberry Pi 5 ---")
    
    if not os.path.exists(MODEL_SOURCE):
        print(f"Error: Source model not found at {MODEL_SOURCE}")
        return

    # 1. Export FP32 ONNX
    print(f"Loading model: {MODEL_SOURCE}")
    model = YOLO(MODEL_SOURCE)
    
    print("Step 1: Exporting to standard FP32 ONNX...")
    exported_path = model.export(format="onnx", imgsz=640, opset=12)
    
    print(f"FP32 Export complete: {exported_path}")

    # 2. Quantize Manually
    print("Step 2: Applying Dynamic INT8 Quantization...")
    
    quantized_path = os.path.join(TARGET_DIR, FINAL_OUTPUT_NAME)
    
    try:
        quantize_dynamic(
            model_input=exported_path,
            model_output=quantized_path,
            weight_type=QuantType.QUInt8 
        )
        print(f"Quantization complete.")
    except Exception as e:
        print(f"❌ Quantization Failed: {e}")
        return

    # 3. Cleanup
    print("Step 3: Cleaning up intermediate files...")
    if os.path.exists(str(exported_path)):
        try:
            os.remove(str(exported_path))
            print(f"Deleted intermediate FP32 file: {exported_path}")
        except Exception as e:
            print(f"Warning: Could not delete intermediate file: {e}")

    print(f"✅ SUCCESS: Optimized model saved as {quantized_path}")
    print(f"You can now transfer '{quantized_path}' to your Raspberry Pi.")

if __name__ == "__main__":
    main()
