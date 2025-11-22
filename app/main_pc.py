import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import sys

# Ensure we can import from utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.img_processing import preprocess_image

def main():
    print("--- Pufferfish Detection System (PC Prototype) ---")
    
    # 1. Load Model
    model_path = "yolo11m_int8.onnx"
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found in current directory.")
        print("Please run 'training/train_export_pc.py' first or ensure the model is in the 'app' folder.")
        return

    print(f"Loading model: {model_path}...")
    try:
        # Using Ultralytics wrapper for ONNX Runtime inference
        # This handles loading the ONNX session and NMS post-processing
        model = YOLO(model_path, task="detect")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Setup Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam (ID 0).")
        return

    # Set resolution (optional, can adjust based on camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Starting inference loop. Press 'q' to exit.")
    
    prev_frame_time = 0
    new_frame_time = 0

    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        # 3. Preprocessing (Lab-CLAHE)
        enhanced_frame = preprocess_image(frame)

        # 4. Inference
        # conf=0.50 as requested
        results = model.predict(source=enhanced_frame, conf=0.50, verbose=False)

        # 5. Visualization
        # We can use results[0].plot() or draw manually.
        # Drawing manually gives us more control if needed, but plot() is robust.
        # Let's use plot() for the prototype to ensure clean visualization of classes/conf.
        annotated_frame = results[0].plot()

        # Calculate FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        
        # Draw FPS on frame
        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show comparison (Optional: Show original vs enhanced? Maybe just enhanced with boxes)
        # User asked to "Display result".
        cv2.imshow("Pufferfish Detection (PC Prototype)", annotated_frame)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Program exited.")

if __name__ == "__main__":
    main()
