import cv2
import time
from ultralytics import YOLO
import sys

# Try importing GPIO, handle if running on PC for testing
try:
    from gpiozero import LED
    GPIO_AVAILABLE = True
    led = LED(17) # GPIO Pin 17
    print("GPIO initialized on Pin 17")
except ImportError:
    GPIO_AVAILABLE = False
    print("GPIO library not found (Running on PC?). LED control disabled.")

# Constants
MODEL_PATH = "pufferfish_pi_int8.onnx" # The quantized model
CONF_THRESHOLD = 0.50 # Slightly lower threshold for quantized model
CLASS_NAMES = {0: 'Pufferfish'}

def main():
    print("--- Pufferfish Detection System (Raspberry Pi Edition) ---")
    
    # 1. Load Model
    try:
        print(f"Loading model: {MODEL_PATH}")
        # Ultralytics handles ONNX loading automatically
        model = YOLO(MODEL_PATH, task='detect') 
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Critical Error loading model: {e}")
        return

    # 2. Setup Video Capture
    # On Pi, 0 usually maps to the connected camera (USB or CSI via libcamera-v4l2)
    video_source = 0
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return

    # Optimize Camera for Pi Performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640) # Match model input size
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("--- Live Monitor Started ---")
    print("Press 'q' to exit.")

    prev_time = 0
    
    while True:
        # 3. Capture
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        t1 = time.time()
        
        # 4. Inference
        # imgsz=640 matches export size
        results = model.predict(source=frame, conf=CONF_THRESHOLD, imgsz=640, verbose=False)
        
        t2 = time.time()
        
        # Metrics
        latency_ms = (t2 - t1) * 1000
        fps = 1 / (t2 - prev_time) if prev_time != 0 else 0
        prev_time = t2
        
        # 5. Logic (GPIO & Visualization)
        display_frame = frame.copy()
        height, width = display_frame.shape[:2]
        
        result = results[0]
        boxes = result.boxes
        
        if len(boxes) > 0:
            # Pufferfish Detected!
            if GPIO_AVAILABLE:
                led.on()
            
            # Visual Alarm
            cv2.rectangle(display_frame, (0, 0), (width, height), (0, 0, 255), 10)
            cv2.putText(display_frame, "PUFFERFISH DETECTED", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            
            # Draw Boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        else:
            # No Detection
            if GPIO_AVAILABLE:
                led.off()

        # Overlay Metrics
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show (Optional on headless Pi, but good for testing)
        cv2.imshow("Pi Monitor", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if GPIO_AVAILABLE:
        led.off()

if __name__ == "__main__":
    main()
