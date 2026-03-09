import os
from dotenv import load_dotenv
from ultralytics import YOLO
from roboflow import Roboflow
from onnxruntime.quantization import quantize_dynamic, QuantType

def main():
    # Load environment variables
    load_dotenv()

    # ---------------------------------------------------------
    # 1. Configuration
    # ---------------------------------------------------------
    API_KEY = os.getenv("ROBOFLOW_API_KEY", "YOUR_API_KEY")
    WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "baliktespitprojesi")
    PROJECT = os.getenv("ROBOFLOW_PROJECT", "balon-baligi-tespiti-oa6kg")
    VERSION = int(os.getenv("ROBOFLOW_VERSION", 1))

    MODEL_NAME = "yolo11n.pt"  # YOLO11 Nano
    IMG_SIZE = 640
    EPOCHS = 50
    EXPORT_NAME = "pufferfish_yolo11n"

    print(f"Starting pipeline for {MODEL_NAME}...")

    # ---------------------------------------------------------
    # 2. Download Dataset from Roboflow
    # ---------------------------------------------------------
    if API_KEY == "YOUR_API_KEY":
        print("WARNING: Roboflow API Key not set. Assuming dataset is already local or skipping download.")
        # You might want to hardcode a path if you have local data,
        # but for this script we'll assume the user needs to set it.
        dataset_location = os.path.join(os.getcwd(), "dataset") # Default fallback
    else:
        print("Downloading dataset from Roboflow...")
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace(WORKSPACE).project(PROJECT)
        dataset = project.version(VERSION).download("yolov8")
        dataset_location = dataset.location
        print(f"Dataset downloaded to: {dataset_location}")

    # ---------------------------------------------------------
    # 3. Train YOLO11n
    # ---------------------------------------------------------
    # Load the model
    model = YOLO(MODEL_NAME)

    # Train
    # Note: 'data' arg usually points to data.yaml inside the downloaded dataset
    data_yaml_path = os.path.join(dataset_location, "data.yaml")

    if not os.path.exists(data_yaml_path):
         print(f"Error: data.yaml not found at {data_yaml_path}. Please check dataset download.")
         return

    print("Starting training...")
    results = model.train(
        data=data_yaml_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        plots=True,
    )

    # ---------------------------------------------------------
    # 4. Export to ONNX
    # ---------------------------------------------------------
    print("Exporting to ONNX...")
    # Export returns the path to the exported file
    onnx_path = model.export(format="onnx", dynamic=False, imgsz=IMG_SIZE)
    print(f"Model exported to: {onnx_path}")

    # ---------------------------------------------------------
    # 5. INT8 Quantization (CPU Optimization)
    # ---------------------------------------------------------
    print("Applying INT8 Dynamic Quantization...")

    quantized_model_path = onnx_path.replace(".onnx", "_int8.onnx")

    quantize_dynamic(
        model_input=onnx_path,
        model_output=quantized_model_path,
        weight_type=QuantType.QUInt8
    )

    print(f"Quantization complete! Optimized model saved to: {quantized_model_path}")
    print("Deploy this file to your Raspberry Pi 5.")

if __name__ == "__main__":
    main()
