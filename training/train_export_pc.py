from ultralytics import YOLO
from onnxruntime.quantization import quantize_dynamic, QuantType
import os
import shutil
import yaml

def fix_dataset_yaml(yaml_path, dataset_root):
    """
    Rewrites the data.yaml file to use absolute paths for 'path', 'train', and 'val'.
    This fixes issues where relative paths in downloaded datasets don't resolve correctly.
    """
    print(f"Checking and fixing YAML at: {yaml_path}")
    
    if not os.path.exists(yaml_path):
        print(f"Error: YAML file not found at {yaml_path}")
        return False

    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Update paths to be absolute
        data['path'] = dataset_root
        data['train'] = os.path.join(dataset_root, "train", "images")
        data['val'] = os.path.join(dataset_root, "valid", "images")
        
        # If there's a test set
        if os.path.exists(os.path.join(dataset_root, "test", "images")):
            data['test'] = os.path.join(dataset_root, "test", "images")

        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
            
        print("YAML file updated with absolute paths.")
        print(f"  path: {data.get('path')}")
        print(f"  train: {data.get('train')}")
        print(f"  val: {data.get('val')}")
        return True
        
    except Exception as e:
        print(f"Error updating YAML: {e}")
        return False

def main():
    print("--- Phase 2: YOLO11m Training & Export (PC - GPU) ---")

    # 1. Configuration
    dataset_root = r"c:/AI/Balik_Projesi_Antigravity/dataset"
    dataset_yaml = r"c:/AI/Balik_Projesi_Antigravity/dataset/data.yaml"
    output_model_name = "yolo11m_pufferfish"

    # 2. Fix YAML Paths
    if not fix_dataset_yaml(dataset_yaml, dataset_root):
        print("Aborting due to YAML error.")
        return

    # 3. Training
    print("\n[1/3] Starting YOLO11m Training...")
    model = YOLO("yolo11m.pt")  # YOLO11 Medium - balanced speed/accuracy

    # Train the model
    results = model.train(
        data=dataset_yaml,
        epochs=50,
        imgsz=640,
        batch=8,
        mosaic=1.0,
        device=0,  # GPU
        project="runs/detect",
        name=output_model_name,
        exist_ok=True
    )

    # 4. Export to ONNX
    print("\n[2/3] Exporting to ONNX...")
    onnx_path = model.export(format="onnx", dynamic=True)
    
    if not os.path.exists(str(onnx_path)):
        onnx_path = os.path.join(results.save_dir, "weights", "best.onnx")
    
    print(f"ONNX Model exported to: {onnx_path}")

    # 5. INT8 Quantization for Raspberry Pi
    print("\n[3/3] Applying INT8 Quantization for Pi...")
    quantized_model_path = "yolo11m_int8.onnx"
    
    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=quantized_model_path,
        weight_type=QuantType.QUInt8
    )
    
    print(f"Quantized INT8 Model saved to: {os.path.abspath(quantized_model_path)}")
    
    # Copy to app directory
    app_dir = os.path.join(os.path.dirname(__file__), "../app")
    if os.path.exists(app_dir):
        dest_path = os.path.join(app_dir, "yolo11m_int8.onnx")
        shutil.copy(quantized_model_path, dest_path)
        print(f"Model copied to: {dest_path}")

    print("\n--- Process Complete ---")
    print(f"Trained model: runs/detect/{output_model_name}/weights/best.pt")
    print(f"Quantized model: {quantized_model_path}")

if __name__ == "__main__":
    main()
