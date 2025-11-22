import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import onnxruntime as ort
import os
import time
from utils.img_processing import preprocess_image

# Constants
MODEL_PATH = "yolo11m_int8.onnx"  # Default model path
CONF_THRESHOLD = 0.50
IOU_THRESHOLD = 0.45
CLASS_NAMES = ['Pufferfish']

class YOLOv8_ONNX:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
        except Exception as e:
            print(f"CUDA provider failed or not available, falling back to CPU. Error: {e}")
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        # Handle dynamic axes where shape might be strings
        if isinstance(self.input_shape[2], int):
            self.img_size = self.input_shape[2]
        else:
            self.img_size = 640 # Default to training size

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

    def infer(self, image, conf_threshold=0.50):
        input_tensor, scale, (pad_top, pad_left) = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        
        # Post-process
        predictions = np.squeeze(outputs[0]).T
        
        boxes = []
        scores = []
        class_ids = []
        
        # Get max confidence for each row
        max_scores = np.max(predictions[:, 4:], axis=1)
        keep_indices = max_scores > conf_threshold
        
        filtered_preds = predictions[keep_indices]
        filtered_scores = max_scores[keep_indices]
        
        if len(filtered_preds) == 0:
            return [], [], []

        for i, pred in enumerate(filtered_preds):
            classes_scores = pred[4:]
            class_id = np.argmax(classes_scores)
            
            cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]
            
            x = cx - w / 2
            y = cy - h / 2
            
            x = (x - pad_left) / scale
            y = (y - pad_top) / scale
            w = w / scale
            h = h / scale
            
            boxes.append([int(x), int(y), int(w), int(h)])
            scores.append(float(filtered_scores[i]))
            class_ids.append(class_id)
            
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, IOU_THRESHOLD)
        
        final_boxes = []
        final_scores = []
        final_class_ids = []
        
        if len(indices) > 0:
            for i in indices.flatten():
                final_boxes.append(boxes[i])
                final_scores.append(scores[i])
                final_class_ids.append(class_ids[i])
                
        return final_boxes, final_scores, final_class_ids

class PufferfishTester:
    def __init__(self, root):
        self.root = root
        self.root.title("Pufferfish Lab 🐡")
        self.root.geometry("500x400")
        self.root.configure(bg="#2E2E2E")
        
        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure colors
        bg_color = "#2E2E2E"
        fg_color = "#FFFFFF"
        accent_color = "#4CAF50"
        button_color = "#2196F3"
        
        self.style.configure("TFrame", background=bg_color)
        self.style.configure("TLabel", background=bg_color, foreground=fg_color, font=("Segoe UI", 10))
        self.style.configure("Header.TLabel", font=("Segoe UI", 18, "bold"), foreground=accent_color)
        self.style.configure("Status.TLabel", font=("Segoe UI", 9), foreground="#AAAAAA")
        
        self.style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=10, borderwidth=0)
        self.style.map("TButton",
            background=[('active', '#1976D2'), ('!disabled', button_color)],
            foreground=[('!disabled', 'white')]
        )
        
        self.style.configure("Horizontal.TScale", background=bg_color, troughcolor="#404040", sliderlength=20)

        # --- Layout ---
        
        # Main Container
        main_frame = ttk.Frame(root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        ttk.Label(header_frame, text="Pufferfish Detection Lab", style="Header.TLabel").pack(side=tk.LEFT)
        ttk.Label(header_frame, text="v1.0", style="Status.TLabel").pack(side=tk.RIGHT, anchor="s")
        
        # Configuration Section
        config_frame = ttk.LabelFrame(main_frame, text="  Settings  ", padding=15)
        config_frame.pack(fill=tk.X, pady=10)
        
        # Custom styling for LabelFrame (tkinter standard widget works better for borders here)
        config_frame.configure(style="TFrame") # Fallback, but let's use standard tk for frame border color if needed
        # Actually ttk LabelFrame is fine, just needs style.
        self.style.configure("TLabelframe", background=bg_color, foreground=fg_color, bordercolor="#555555")
        self.style.configure("TLabelframe.Label", background=bg_color, foreground=accent_color, font=("Segoe UI", 10, "bold"))

        # Slider
        slider_container = ttk.Frame(config_frame)
        slider_container.pack(fill=tk.X)
        
        self.conf_var = tk.DoubleVar(value=0.50)
        
        top_row = ttk.Frame(slider_container)
        top_row.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(top_row, text="Sensitivity (Confidence Threshold)").pack(side=tk.LEFT)
        self.val_label = ttk.Label(top_row, text="50%", foreground=accent_color, font=("Segoe UI", 10, "bold"))
        self.val_label.pack(side=tk.RIGHT)
        
        self.scale = ttk.Scale(slider_container, from_=0.1, to=1.0, variable=self.conf_var, command=self.update_label, orient=tk.HORIZONTAL)
        self.scale.pack(fill=tk.X)
        
        ttk.Label(slider_container, text="Lower value = More detections (but maybe false positives)", style="Status.TLabel").pack(anchor="w", pady=(5, 0))

        # Actions Section
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill=tk.BOTH, expand=True, pady=20)
        
        self.btn_select = ttk.Button(action_frame, text="📂  Open Image or Video", command=self.select_file, cursor="hand2")
        self.btn_select.pack(fill=tk.X, ipady=5)
        
        ttk.Label(action_frame, text="Supports: JPG, PNG, MP4, AVI", style="Status.TLabel").pack(pady=5)

        # Status Bar
        self.status_var = tk.StringVar(value="Ready to process.")
        status_bar = ttk.Label(root, textvariable=self.status_var, style="Status.TLabel", padding=(10, 5), background="#252525")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Model Loading
        self.load_model()

    def update_label(self, value):
        val = float(value)
        self.val_label.config(text=f"{int(val*100)}%")

    def load_model(self):
        # Try to load model from app dir or parent dir
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "yolo11m_int8.onnx"),
            os.path.join(os.path.dirname(__file__), "../yolo11m_int8.onnx"),
            "yolo11m_int8.onnx"
        ]
        
        self.model_path = None
        for p in possible_paths:
            if os.path.exists(p):
                self.model_path = p
                break
                
        if not self.model_path:
            self.status_var.set("Error: Model not found!")
            messagebox.showerror("Error", "Model file 'yolo11m_int8.onnx' not found!")
            return
            
        self.status_var.set(f"Loading model: {os.path.basename(self.model_path)}...")
        self.root.update()
        
        try:
            self.detector = YOLOv8_ONNX(self.model_path)
            self.status_var.set("Model loaded successfully. Ready.")
        except Exception as e:
            self.status_var.set("Error loading model.")
            messagebox.showerror("Error", f"Failed to load model: {e}")

    def select_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Image or Video",
            filetypes=[("Media Files", "*.jpg *.jpeg *.png *.mp4 *.avi *.mov"), ("All Files", "*.*")]
        )
        
        if file_path:
            self.process_file(file_path)

    def process_file(self, file_path):
        self.status_var.set(f"Processing: {os.path.basename(file_path)}")
        self.root.update()
        
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png']:
            self.process_image(file_path)
        elif ext in ['.mp4', '.avi', '.mov']:
            self.process_video(file_path)
        else:
            messagebox.showwarning("Warning", "Unsupported file format.")
            self.status_var.set("Ready.")

    def draw_detections(self, frame, boxes, scores, class_ids):
        for box, score, class_id in zip(boxes, scores, class_ids):
            x, y, w, h = box
            label = f"{CLASS_NAMES[class_id]}: {score:.2f}"
            color = (0, 0, 255) # Red for pufferfish
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def process_image(self, file_path):
        try:
            image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            image = None
            
        if image is None:
            messagebox.showerror("Error", "Could not read image.")
            self.status_var.set("Error reading file.")
            return
            
        processed = preprocess_image(image)
        conf_threshold = self.conf_var.get()
        
        start = time.time()
        boxes, scores, class_ids = self.detector.infer(processed, conf_threshold)
        end = time.time()
        
        self.status_var.set(f"Inference: {(end-start)*1000:.1f}ms | Detections: {len(boxes)}")
        
        result = self.draw_detections(image.copy(), boxes, scores, class_ids)
        
        cv2.imshow(f"Result (Conf: {int(conf_threshold*100)}%)", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.status_var.set("Ready.")

    def process_video(self, file_path):
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open video.")
            return
            
        self.status_var.set("Playing video... Press 'q' to stop.")
        print("Press 'q' to stop video.")
        
        prev_time = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            processed = preprocess_image(frame)
            conf_threshold = self.conf_var.get()
            
            boxes, scores, class_ids = self.detector.infer(processed, conf_threshold)
            result = self.draw_detections(frame, boxes, scores, class_ids)
            
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time
            cv2.putText(result, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Video Detection (Press 'q' to exit)", result)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        self.status_var.set("Ready.")

if __name__ == "__main__":
    root = tk.Tk()
    # Import ttk here if not already imported globally, but usually it's standard
    from tkinter import ttk 
    app = PufferfishTester(root)
    root.mainloop()
