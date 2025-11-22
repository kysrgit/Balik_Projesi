import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import onnxruntime as ort
import os
import time
import threading
import shutil
from datetime import datetime
from utils.img_processing import preprocess_image
from PIL import Image

# Configuration
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

# Constants
MODEL_PATH = "yolo11m_int8.onnx"
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
        
        if isinstance(self.input_shape[2], int):
            self.img_size = self.input_shape[2]
        else:
            self.img_size = 640

    def preprocess(self, image):
        h, w = image.shape[:2]
        scale = self.img_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))
        
        top = (self.img_size - new_h) // 2
        bottom = self.img_size - new_h - top
        left = (self.img_size - new_w) // 2
        right = self.img_size - new_w - left
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        blob = cv2.dnn.blobFromImage(padded, 1/255.0, (self.img_size, self.img_size), swapRB=True, crop=False)
        return blob, scale, (top, left)

    def infer(self, image, conf_threshold=0.50):
        input_tensor, scale, (pad_top, pad_left) = self.preprocess(image)
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        predictions = np.squeeze(outputs[0]).T
        
        boxes = []
        scores = []
        class_ids = []
        
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

class BatchProcessorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Pufferfish Batch Validator | Deep Scan")
        self.geometry("900x600")
        
        # Grid Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # --- Sidebar ---
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(4, weight=1)
        
        self.logo_label = ctk.CTkLabel(self.sidebar, text="🐡 Pufferfish\nDetection System", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        self.btn_folder = ctk.CTkButton(self.sidebar, text="Select Folder", command=self.select_folder, height=40, font=ctk.CTkFont(size=14))
        self.btn_folder.grid(row=1, column=0, padx=20, pady=20)
        
        self.lbl_conf = ctk.CTkLabel(self.sidebar, text="Sensitivity: 50%", anchor="w")
        self.lbl_conf.grid(row=2, column=0, padx=20, pady=(10, 0))
        
        self.slider = ctk.CTkSlider(self.sidebar, from_=0.1, to=1.0, number_of_steps=18, command=self.update_slider_label)
        self.slider.set(0.50)
        self.slider.grid(row=3, column=0, padx=20, pady=(0, 20))
        
        self.btn_start = ctk.CTkButton(self.sidebar, text="START SCAN", command=self.start_batch, height=50, fg_color="#00C853", hover_color="#009624", font=ctk.CTkFont(size=16, weight="bold"), state="disabled")
        self.btn_start.grid(row=5, column=0, padx=20, pady=20)

        # --- Main Area ---
        self.main_area = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_area.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        
        # Status Card
        self.status_frame = ctk.CTkFrame(self.main_area)
        self.status_frame.pack(fill="x", pady=(0, 20))
        
        self.lbl_status_title = ctk.CTkLabel(self.status_frame, text="Current Status", font=ctk.CTkFont(size=12, weight="bold"), text_color="gray")
        self.lbl_status_title.pack(anchor="w", padx=15, pady=(10, 0))
        
        self.lbl_status = ctk.CTkLabel(self.status_frame, text="Ready to Scan", font=ctk.CTkFont(size=16))
        self.lbl_status.pack(anchor="w", padx=15, pady=(0, 10))
        
        self.progress = ctk.CTkProgressBar(self.status_frame)
        self.progress.pack(fill="x", padx=15, pady=(0, 15))
        self.progress.set(0)
        
        # Metrics Dashboard
        self.metrics_frame = ctk.CTkFrame(self.main_area, fg_color="transparent")
        self.metrics_frame.pack(fill="x", pady=(0, 20))
        
        self.card_total = self.create_metric_card(self.metrics_frame, "Total Files", "0", "#2196F3")
        self.card_total.pack(side="left", expand=True, fill="both", padx=(0, 10))
        
        self.card_detected = self.create_metric_card(self.metrics_frame, "Pufferfish Detected", "0", "#F44336")
        self.card_detected.pack(side="left", expand=True, fill="both", padx=5)
        
        self.card_clean = self.create_metric_card(self.metrics_frame, "Safe / Clean", "0", "#4CAF50")
        self.card_clean.pack(side="left", expand=True, fill="both", padx=(10, 0))
        
        # Log Area
        self.log_box = ctk.CTkTextbox(self.main_area, font=ctk.CTkFont(family="Consolas", size=12))
        self.log_box.pack(fill="both", expand=True)
        self.log_box.insert("0.0", "System initialized.\nWaiting for folder selection...\n")
        
        # State
        self.folder_path = None
        self.model = None
        self.is_running = False
        self.stats = {"total": 0, "detected": 0, "clean": 0}
        
        # Load Model
        self.load_model()

    def create_metric_card(self, parent, title, value, color):
        frame = ctk.CTkFrame(parent)
        ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=12, weight="bold"), text_color="gray").pack(pady=(10, 0))
        lbl_value = ctk.CTkLabel(frame, text=value, font=ctk.CTkFont(size=24, weight="bold"), text_color=color)
        lbl_value.pack(pady=(0, 10))
        frame.value_label = lbl_value # Store reference
        return frame

    def update_slider_label(self, value):
        self.lbl_conf.configure(text=f"Sensitivity: {int(value*100)}%")

    def log(self, message):
        self.log_box.insert("end", f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
        self.log_box.see("end")

    def load_model(self):
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "yolo11m_int8.onnx"),
            os.path.join(os.path.dirname(__file__), "../yolo11m_int8.onnx"),
            "yolo11m_int8.onnx"
        ]
        model_path = next((p for p in possible_paths if os.path.exists(p)), None)
        
        if model_path:
            try:
                self.model = YOLOv8_ONNX(model_path)
                self.log(f"Model loaded: {os.path.basename(model_path)}")
            except Exception as e:
                self.log(f"Error loading model: {e}")
                messagebox.showerror("Error", f"Model load failed: {e}")
        else:
            self.log("Model not found!")
            messagebox.showerror("Error", "Model not found!")

    def select_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.folder_path = folder
            self.log(f"Selected folder: {folder}")
            self.btn_start.configure(state="normal")
            self.lbl_status.configure(text="Ready to Scan")

    def start_batch(self):
        if not self.folder_path or not self.model:
            return
            
        self.is_running = True
        self.btn_start.configure(state="disabled")
        self.btn_folder.configure(state="disabled")
        self.slider.configure(state="disabled")
        
        # Reset stats
        self.stats = {"total": 0, "detected": 0, "clean": 0}
        self.update_metrics()
        
        threading.Thread(target=self.run_process, daemon=True).start()

    def update_metrics(self):
        self.card_total.value_label.configure(text=str(self.stats["total"]))
        self.card_detected.value_label.configure(text=str(self.stats["detected"]))
        self.card_clean.value_label.configure(text=str(self.stats["clean"]))

    def run_process(self):
        try:
            files = [f for f in os.listdir(self.folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov'))]
            total_files = len(files)
            self.stats["total"] = total_files
            self.update_metrics()
            
            if total_files == 0:
                self.log("No supported files found.")
                self.reset_ui()
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join(self.folder_path, f"Batch_Results_{timestamp}")
            pos_dir = os.path.join(out_dir, "Positive_Detections")
            neg_dir = os.path.join(out_dir, "Negatives_Clean")
            
            os.makedirs(pos_dir, exist_ok=True)
            os.makedirs(neg_dir, exist_ok=True)
            
            self.log(f"Starting scan...")
            conf_threshold = self.slider.get()
            
            for i, filename in enumerate(files):
                if not self.is_running: break
                
                filepath = os.path.join(self.folder_path, filename)
                ext = os.path.splitext(filename)[1].lower()
                
                self.lbl_status.configure(text=f"Scanning: {filename}")
                self.progress.set((i+1) / total_files)
                
                if ext in ['.jpg', '.jpeg', '.png']:
                    self.process_image(filepath, filename, pos_dir, neg_dir, conf_threshold)
                elif ext in ['.mp4', '.avi', '.mov']:
                    self.process_video(filepath, filename, pos_dir, neg_dir, conf_threshold)
                
                self.update_metrics()
            
            self.log("Scan complete!")
            self.lbl_status.configure(text="Scan Complete")
            os.startfile(out_dir)
            
        except Exception as e:
            self.log(f"Critical Error: {e}")
        finally:
            self.reset_ui()

    def reset_ui(self):
        self.is_running = False
        self.btn_start.configure(state="normal")
        self.btn_folder.configure(state="normal")
        self.slider.configure(state="normal")
        self.progress.set(0)

    def process_image(self, filepath, filename, pos_dir, neg_dir, conf):
        try:
            img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None: return
            
            processed = preprocess_image(img)
            boxes, scores, class_ids = self.model.infer(processed, conf)
            
            if len(boxes) > 0:
                self.stats["detected"] += 1
                for box, score, cid in zip(boxes, scores, class_ids):
                    x, y, w, h = box
                    label = f"{CLASS_NAMES[cid]}: {score:.2f}"
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                out_path = os.path.join(pos_dir, filename)
                cv2.imencode(os.path.splitext(filename)[1], img)[1].tofile(out_path)
                self.log(f"FOUND: {filename}")
            else:
                self.stats["clean"] += 1
                out_path = os.path.join(neg_dir, filename)
                shutil.copy2(filepath, out_path)
                
        except Exception as e:
            self.log(f"Error: {e}")

    def process_video(self, filepath, filename, pos_dir, neg_dir, conf):
        try:
            cap = cv2.VideoCapture(filepath)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            has_detection = False
            temp_out_path = os.path.join(pos_dir, f"temp_{filename}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_out_path, fourcc, fps, (width, height))
            
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                processed = preprocess_image(frame)
                boxes, scores, class_ids = self.model.infer(processed, conf)
                
                if len(boxes) > 0:
                    has_detection = True
                    for box, score, cid in zip(boxes, scores, class_ids):
                        x, y, w, h = box
                        label = f"{CLASS_NAMES[cid]}: {score:.2f}"
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                out.write(frame)
                
            cap.release()
            out.release()
            
            if has_detection:
                self.stats["detected"] += 1
                final_path = os.path.join(pos_dir, filename)
                if os.path.exists(final_path): os.remove(final_path)
                os.rename(temp_out_path, final_path)
                self.log(f"FOUND (Video): {filename}")
            else:
                self.stats["clean"] += 1
                os.remove(temp_out_path)
                out_path = os.path.join(neg_dir, filename)
                shutil.copy2(filepath, out_path)
                
        except Exception as e:
            self.log(f"Error: {e}")

if __name__ == "__main__":
    app = BatchProcessorApp()
    app.mainloop()
