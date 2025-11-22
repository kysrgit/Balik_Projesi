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
        # We explicitly set 'path' to the dataset root
        data['path'] = dataset_root
        # 'train' and 'val' are usually relative to 'path' or absolute. 
        # Setting them to 'train/images' and 'valid/images' works if 'path' is correct.
        # But to be safe and "blindly trust" the structure as requested:
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
    print("--- Phase 2: Model Training & Export (PC - Local Data) ---")

    # 1. Configuration
    # Hardcoded path as requested
    dataset_root = r"c:/AI/Balik_Projesi_Antigravity/dataset"
    dataset_yaml = r"c:/AI/Balik_Projesi_Antigravity/dataset/data.yaml"

    # 2. Fix YAML Paths
    if not fix_dataset_yaml(dataset_yaml, dataset_root):
        print("Aborting due to YAML error.")
        return

    # 3. Training
    print("\n[1/3] Starting YOLO11m Training...")
    # Load a model
    model = YOLO("yolo11m.pt")  # load a pretrained model

    # Train the model
    # batch=4 as requested for memory safety
    results = model.train(
        data=dataset_yaml,
        epochs=30,
        imgsz=640,
        batch=4,
        mosaic=1.0,
        device=0, # Force GPU for PC prototype validation
        project="runs/detect",
        name="yolo11m_pufferfish",
        exist_ok=True
    )

    # 4. Export to ONNX
    print("\n[2/3] Exporting to ONNX...")
    # Export the best model
    success = model.export(format="onnx", dynamic=True)
    
    onnx_path = str(success)
    
    if not os.path.exists(onnx_path):
        # Fallback path construction
        onnx_path = os.path.join(results.save_dir, "weights", "best.onnx")
    
    print(f"ONNX Model exported to: {onnx_path}")

    # 5. INT8 Quantization
    print("\n[3/3] Applying INT8 Quantization...")
    quantized_model_path = "yolo11m_int8.onnx" # Save in current dir as requested
    
    # If we want to save it exactly where the original was, we can, but user asked for "yolo11m_int8.onnx"
    # Let's save it in the current directory for easy access by the app
    
    quantize_dynamic(
        model_input=onnx_path,
        model_output=quantized_model_path,
        weight_type=QuantType.QUInt8
    )
    
    print(f"Quantized INT8 Model saved to: {os.path.abspath(quantized_model_path)}")
    
    # Also copy to app directory if it exists, just in case
    app_dir = os.path.join(os.path.dirname(__file__), "../app")
    if os.path.exists(app_dir):
        shutil.copy(quantized_model_path, os.path.join(app_dir, "yolo11m_int8.onnx"))
        print(f"Model copied to app directory: {app_dir}")

    print("--- Process Complete ---")

if __name__ == "__main__":
    main()
