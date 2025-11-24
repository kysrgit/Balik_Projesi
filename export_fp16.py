from ultralytics import YOLO
import shutil

# Load the trained model
model_path = "runs/detect/yolo11m_pufferfish/weights/best.pt"
model = YOLO(model_path)

print(f"Loading trained model from: {model_path}")

# Export to ONNX with FP16 (half precision)
print("Exporting to ONNX FP16 format...")
model.export(
    format="onnx",
    imgsz=640,
    half=True,  # FP16 for speed
    opset=12
)

# The exported file will be in the same directory as best.pt
exported_path = "runs/detect/yolo11m_pufferfish/weights/best.onnx"
target_path = "yolo11m_pufferfish_fp16.onnx"
app_target_path = "app/yolo11m_pufferfish_fp16.onnx"

print(f"Copying {exported_path} to {target_path}")
shutil.copy(exported_path, target_path)

print(f"Copying to app directory: {app_target_path}")
shutil.copy(exported_path, app_target_path)

print("FP16 Export Complete!")
print(f"Model saved as: {app_target_path}")
