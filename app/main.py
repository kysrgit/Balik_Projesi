import cv2
import time
import os
import numpy as np
import onnxruntime as ort
from utils.img_processing import apply_lab_clahe

# Constants
MODEL_PATH = "pufferfish_yolo11n_int8.onnx" # Ensure this matches your exported filename
CONF_THRESHOLD = 0.55
IOU_THRESHOLD = 0.45
CLASS_NAMES = ['Pufferfish'] # Update if you have more classes
DETECTION_DIR = "detections"

class YOLOv8_ONNX:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.img_size = self.input_shape[2] # Assuming square input [1, 3, 640, 640]

    def preprocess(self, image):
        # Resize and pad
        h, w = image.shape[:2]
        scale = self.img_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))
        
        # Padding
        top = (self.img_size - new_h) // 2
        bottom = self.img_size - new_h - top
        left = (self.img_size - new_w) // 2
        right = self.img_size - new_w - left
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # Normalize
        blob = cv2.dnn.blobFromImage(padded, 1/255.0, (self.img_size, self.img_size), swapRB=True, crop=False)
        return blob, scale, (top, left)

    def infer(self, image):
        input_tensor, scale, (pad_top, pad_left) = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        
        # Post-process
        # Output shape is usually [1, 4+num_classes, 8400]
        # We need to transpose to [1, 8400, 4+num_classes]
        predictions = np.squeeze(outputs[0]).T
        
        boxes = []
        scores = []
        class_ids = []
        
        # Filter by confidence
        # predictions format: [x, y, w, h, class1_conf, class2_conf, ...]
        
        # Get max confidence for each row
        max_scores = np.max(predictions[:, 4:], axis=1)
        keep_indices = max_scores > CONF_THRESHOLD
        
        filtered_preds = predictions[keep_indices]
        filtered_scores = max_scores[keep_indices]
        
        if len(filtered_preds) == 0:
            return [], [], []

        for i, pred in enumerate(filtered_preds):
            # Get class ID
            classes_scores = pred[4:]
            class_id = np.argmax(classes_scores)
            
            # Box coordinates (center_x, center_y, w, h)
            cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]
            
            # Convert to top-left x, y
            x = cx - w / 2
            y = cy - h / 2
            
            # Adjust for padding and scaling
            x = (x - pad_left) / scale
            y = (y - pad_top) / scale
            w = w / scale
            h = h / scale
            
            boxes.append([int(x), int(y), int(w), int(h)])
            scores.append(float(filtered_scores[i]))
            class_ids.append(class_id)
            
        # NMS
        indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, IOU_THRESHOLD)
        
        final_boxes = []
        final_scores = []
        final_class_ids = []
        
        if len(indices) > 0:
            for i in indices.flatten():
                final_boxes.append(boxes[i])
                final_scores.append(scores[i])
                final_class_ids.append(class_ids[i])
                
        return final_boxes, final_scores, final_class_ids

def main():
    # Create detections directory
    if not os.path.exists(DETECTION_DIR):
        os.makedirs(DETECTION_DIR)

    # Initialize Camera
    # On RPi, 0 usually maps to the first camera. 
    # Ensure libcamera-compatibility layer is active or use specific index.
    cap = cv2.VideoCapture(0) 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Initialize Model
    print(f"Loading model from {MODEL_PATH}...")
    try:
        detector = YOLOv8_ONNX(MODEL_PATH)
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Make sure you have run the training/export script and the .onnx file exists.")
        return

    print("Starting Inference Loop. Press 'q' to exit.")
    
    prev_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # 1. Preprocessing (Lab-CLAHE)
        processed_frame = apply_lab_clahe(frame)
        
        # 2. Inference
        boxes, scores, class_ids = detector.infer(processed_frame)
        
        # 3. Draw & Log
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        for box, score, class_id in zip(boxes, scores, class_ids):
            x, y, w, h = box
            label = f"{CLASS_NAMES[class_id]}: {score:.2f}"
            
            # Draw Box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Snapshot Logic
            if CLASS_NAMES[class_id] == "Pufferfish" and score > CONF_THRESHOLD:
                timestamp = int(time.time())
                filename = f"{DETECTION_DIR}/puffer_{timestamp}_{score:.2f}.jpg"
                # Save the ORIGINAL frame (or processed if preferred, usually original is better for record)
                # We'll save the one with boxes for quick review, or clean one? 
                # User said "Save a snapshot". Usually implies the detection event. 
                # I'll save the frame with boxes to verify what was seen.
                cv2.imwrite(filename, frame)
                # print(f"Saved detection: {filename}") # Optional logging

        # Display
        cv2.imshow("Pufferfish Detection System", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
