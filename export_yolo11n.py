from ultralytics import YOLO

# Load standard YOLO11n model (COCO pretrained - knows 'person')
model = YOLO("yolo11n.pt")

# Export to ONNX
model.export(format="onnx", imgsz=640, opset=12)
