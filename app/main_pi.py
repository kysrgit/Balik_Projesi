import cv2
import time
import os
import sys
from datetime import datetime
from ultralytics import YOLO

# Try importing GPIO
try:
    from gpiozero import LED
    GPIO_AVAILABLE = True
    led = LED(17) # GPIO Pin 17
    print("✅ GPIO initialized on Pin 17")
except ImportError:
    GPIO_AVAILABLE = False
    print("⚠️ GPIO library not found. LED control disabled.")

# Constants
MODEL_PATH = "app/pufferfish_pi_int8.onnx" # The quantized model
CONF_THRESHOLD = 0.60
DETECTION_DIR = "detections"
# 🛡️ SECURITY: Limit detection files to prevent disk exhaustion on Pi
MAX_DETECTION_FILES = 100

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def cleanup_old_detections(directory, max_files):
    """🛡️ SECURITY: Remove oldest detection files to prevent disk exhaustion."""
    try:
        files = sorted(
            [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.jpg')],
            key=os.path.getmtime
        )
        while len(files) >= max_files:
            os.remove(files.pop(0))
    except OSError:
        pass  # Fail silently - don't crash detection on cleanup error

def main():
    print("--- Pufferfish Detection System (STABLE HEADLESS MODE) ---")
    ensure_dir(DETECTION_DIR)
    
    # 1. Load Model
    try:
        print(f"Loading model: {MODEL_PATH}")
        model = YOLO(MODEL_PATH, task='detect') 
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Critical Error loading model: {e}")
        return

    # 2. Setup Video Capture (V4L2 Optimized)
    print("📷 Initializing Camera (V4L2)...")
    video_source = 0
    # Explicitly use V4L2 backend for Pi
    cap = cv2.VideoCapture(video_source, cv2.CAP_V4L2)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video source {video_source}")
        return

    # Force YUYV format (Critical for Pi Camera Module stability)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # WARM-UP (Critical Fix)
    print("⏳ Sensör başlatılıyor, lütfen bekleyin (2 saniye)...")
    time.sleep(2)
    print("✅ Kamera Hazır!")

    print("--- Monitoring Started (Press Ctrl+C to stop) ---")

    prev_time = 0
    last_save_time = 0
    
    try:
        while True:
            # 3. Capture
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                time.sleep(0.1) # Short wait before retry
                continue

            t1 = time.time()
            
            # 4. Inference
            results = model.predict(source=frame, conf=CONF_THRESHOLD, imgsz=640, verbose=False)
            
            t2 = time.time()
            
            # Metrics
            fps = 1 / (t2 - prev_time) if prev_time != 0 else 0
            prev_time = t2
            
            # 5. Logic
            result = results[0]
            boxes = result.boxes
            
            status_msg = "NO FISH"
            detection_count = len(boxes)
            
            if detection_count > 0:
                status_msg = "!!! PUFFERFISH DETECTED !!!"
                
                # GPIO Alert
                if GPIO_AVAILABLE:
                    led.on()
                
                # Save Evidence (Max 1 per second)
                current_time = time.time()
                if current_time - last_save_time >= 1.0:
                    # 🛡️ SECURITY: Clean up old files before saving new one
                    cleanup_old_detections(DETECTION_DIR, MAX_DETECTION_FILES)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"fish_{timestamp}.jpg"
                    save_path = os.path.join(DETECTION_DIR, filename)
                    
                    # Draw boxes on the frame before saving
                    save_frame = frame.copy()
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        conf = float(box.conf)
                        cv2.rectangle(save_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(save_frame, f"Pufferfish: {conf:.2f}", (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    cv2.imwrite(save_path, save_frame)
                    print(f"📸 Evidence saved: {save_path}")
                    last_save_time = current_time
            else:
                # No Detection
                if GPIO_AVAILABLE:
                    led.off()

            # Console Log (Every ~1 second)
            if int(t2) % 1 == 0 and int(t2 * 10) % 10 == 0: # Simple throttle
                 print(f"FPS: {fps:.1f} | Status: {status_msg}")

    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
    finally:
        cap.release()
        if GPIO_AVAILABLE:
            led.off()
        print("System Shutdown Complete.")

if __name__ == "__main__":
    main()
