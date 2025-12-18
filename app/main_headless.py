import cv2
import time
import os
import sys
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

# Try importing CLAHE preprocessing
try:
    # Assuming utils is in the same directory or python path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils.img_processing import preprocess_image
    CLAHE_AVAILABLE = True
    print("✅ CLAHE preprocessing module found.")
except ImportError:
    CLAHE_AVAILABLE = False
    print("⚠️ utils.img_processing not found. Using raw frames.")

# Constants
MODEL_PATH = "models/pufferfish_pi_int8.onnx" # The quantized model
CONF_THRESHOLD = 0.60
DETECTION_DIR = "detections"

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    print("--- Pufferfish Detection System (HEADLESS MODE) ---")
    ensure_dir(DETECTION_DIR)
    
    # 1. Load Model
    try:
        print(f"Loading model: {MODEL_PATH}")
        model = YOLO(MODEL_PATH, task='detect') 
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Critical Error loading model: {e}")
        return

    # 2. Setup Video Capture
    video_source = 0
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video source {video_source}")
        return

    # Optimize Camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("--- Monitoring Started (Press Ctrl+C to stop) ---")

    prev_time = 0
    last_save_time = 0
    
    try:
        while True:
            # 3. Capture
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                time.sleep(1)
                continue

            t1 = time.time()
            
            # 4. Preprocess
            if CLAHE_AVAILABLE:
                processed_frame = preprocess_image(frame)
            else:
                processed_frame = frame

            # 5. Inference
            results = model.predict(source=processed_frame, conf=CONF_THRESHOLD, imgsz=640, verbose=False)
            
            t2 = time.time()
            
            # Metrics
            fps = 1 / (t2 - prev_time) if prev_time != 0 else 0
            prev_time = t2
            
            # 6. Logic
            result = results[0]
            boxes = result.boxes
            
            status_msg = "SEARCHING..."
            detection_count = len(boxes)
            
            if detection_count > 0:
                status_msg = "!!! PUFFERFISH DETECTED !!!"
                
                # GPIO Alert
                if GPIO_AVAILABLE:
                    led.on()
                
                # Save Evidence (Max 1 per second)
                current_time = time.time()
                if current_time - last_save_time >= 1.0:
                    # Optimized: reuse current_time instead of calling datetime.now()
                    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(current_time))
                    filename = f"fish_{timestamp}.jpg"
                    save_path = os.path.join(DETECTION_DIR, filename)
                    
                    # Draw boxes on the frame before saving
                    save_frame = processed_frame.copy()
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
                 print(f"FPS: {fps:.1f} | Status: {status_msg} | Detections: {detection_count}")

    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
    finally:
        cap.release()
        if GPIO_AVAILABLE:
            led.off()
        print("System Shutdown Complete.")

if __name__ == "__main__":
    main()
