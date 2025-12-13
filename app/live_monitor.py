import cv2
import time
import torch
from ultralytics import YOLO
import sys

# Constants
MODEL_PATH = "app/yolo11m_pufferfish.pt"
CONF_THRESHOLD = 0.60
CLASS_NAMES = {0: 'Pufferfish'} # YOLO model class mapping

def main():
    print("--- Initializing Pufferfish Detection System (PyTorch) ---")
    
    # 1. Check GPU
    if torch.cuda.is_available():
        device = 'cuda:0'
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU DETECTED: {gpu_name}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("❌ WARNING: GPU NOT DETECTED! Running on CPU (Will be slow)")
        device = 'cpu'

    # 2. Load Model
    try:
        print(f"Loading model from: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        # Force model to GPU
        model.to(device)
        print("✅ Model loaded successfully and moved to GPU")
    except Exception as e:
        print(f"❌ Critical Error loading model: {e}")
        return

    # 3. Setup Video Capture
    video_source = 0
    # Use DirectShow on Windows
    cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return

    # Optimize Camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("--- Live Monitor Started ---")
    print("Press 'q' to exit.")

    prev_time = 0
    
    while True:
        # 4. Capture
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        t1 = time.time()
        
        # 5. Inference
        # verbose=False prevents printing to console every frame
        # half=True uses FP16 for speed (only works on GPU)
        results = model.predict(source=frame, conf=CONF_THRESHOLD, device=device, verbose=False, half=(device!='cpu'))
        
        t2 = time.time()
        
        # Metrics
        latency_ms = (t2 - t1) * 1000
        fps = 1 / (t2 - prev_time) if prev_time != 0 else 0
        prev_time = t2
        
        # Log FPS to console every 30 frames (approx 1 sec)
        if int(fps) > 0 and (int(time.time() * 10) % 10 == 0):
             print(f"FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms")
        
        # 6. Visualization
        # ⚡ PERF: Draw directly on frame instead of copying
        # (frame is discarded after display, copy is unnecessary)
        # Saves ~921KB memory copy per frame (~27MB/s at 30FPS)
        height, width = frame.shape[:2]
        
        # Process detections
        result = results[0]
        boxes = result.boxes
        
        if len(boxes) > 0:
            print("!!! SIGNAL SENT: STOP MOTORS !!!")
            
            # Draw Alarm
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 20)
            cv2.putText(frame, "WARNING: PUFFERFISH DETECTED", (50, height // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            
            # Draw Boxes
            for box in boxes:
                # ⚡ PERF: Get all coordinates at once, convert outside loop would be better
                # but keeping per-box for clarity
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf)
                
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                label = f"Pufferfish: {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Overlay Metrics
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Latency: {latency_ms:.1f}ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        status_text = f"Model: YOLO11m (PyTorch) | GPU: {gpu_name if device!='cpu' else 'OFF'}"
        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.putText(frame, status_text, (width - text_size[0] - 10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("ROV Control Screen - Live Monitor", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
