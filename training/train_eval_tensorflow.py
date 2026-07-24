# Fugaeus (Balon Balığı) Veri Seti TensorFlow / Keras Model Eğitim ve Değerlendirme Betiği
# Bu betik, MobileNetV2 omurgalı TensorFlow/Keras nesne tespit ve sınıflandırma ağını
# dataset/ klasöründe ince ayar (fine-tuning) ile eğitir, optimum eşik arama ile doğrulama
# ve test metriklerini hesaplayıp benchmark_pc/ klasörüne yazar.

import os
import glob
import json
import time
from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# Sabitler ve Dizinler
BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
BENCHMARK_DIR = BASE_DIR / "benchmark_pc"

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
NUM_CLASSES = 1

def load_dataset_split(split_name="train"):
    """
    YOLO etiketli veri seti klasörünü TensorFlow eğitimi için okur ve numpy array formatına dönüştürür.
    """
    img_dir = DATASET_DIR / split_name / "images"
    lbl_dir = DATASET_DIR / split_name / "labels"
    
    if not img_dir.exists():
        img_dir = DATASET_DIR / "images" / split_name
        lbl_dir = DATASET_DIR / "labels" / split_name

    image_paths = sorted(glob.glob(str(img_dir / "*.jpg")) + glob.glob(str(img_dir / "*.png")))
    
    images = []
    boxes = []
    labels = []
    
    for img_path in image_paths:
        base_name = Path(img_path).stem
        lbl_path = lbl_dir / f"{base_name}.txt"
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        has_box = 0
        bbox = [0.0, 0.0, 0.0, 0.0]
        if lbl_path.exists():
            with open(lbl_path, "r") as f:
                lines = f.readlines()
                if len(lines) > 0:
                    parts = lines[0].strip().split()
                    if len(parts) >= 5:
                        has_box = 1
                        bbox = [float(p) for p in parts[1:5]]
                        
        images.append(img_resized / 255.0)
        boxes.append(bbox)
        labels.append(has_box)
        
    return np.array(images, dtype=np.float32), np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.float32)

def build_tensorflow_model():
    """
    MobileNetV2 omurgasına sahip TensorFlow / Keras Sınıflandırma ve Bounding Box Regresyon Modeli.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = True
    for layer in base_model.layers[:80]:
        layer.trainable = False
        
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Sınıflandırma başlığı (Balon Balığı Var/Yok)
    class_output = layers.Dense(1, activation="sigmoid", name="class_output")(x)
    
    # Bounding Box (Kutu Koordinatı) regresyon başlığı (x, y, w, h)
    box_output = layers.Dense(64, activation="relu")(x)
    box_output = layers.Dense(4, activation="sigmoid", name="box_output")(box_output)
    
    model = models.Model(inputs=inputs, outputs=[class_output, box_output])
    return model

def compute_iou(box1, box2):
    """
    [cx, cy, w, h] iki kutu arasındaki IoU (Intersection over Union) değerini hesaplar.
    """
    b1_x1, b1_y1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    b1_x2, b1_y2 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    
    b2_x1, b2_y1 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    b2_x2, b2_y2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2
    
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    b1_area = max(1e-6, (b1_x2 - b1_x1) * (b1_y2 - b1_y1))
    b2_area = max(1e-6, (b2_x2 - b2_x1) * (b2_y2 - b2_y1))
    union_area = b1_area + b2_area - inter_area
    
    return float(inter_area / union_area) if union_area > 0 else 0.0

def train_and_eval_tensorflow():
    """
    TensorFlow modelini eğitir, optimum eşik değerini bulur ve test metriklerini hesaplar.
    """
    print("[+] TensorFlow Veri Seti Yükleniyor...")
    X_train, y_box_train, y_cls_train = load_dataset_split("train")
    X_val, y_box_val, y_cls_val = load_dataset_split("valid")
    X_test, y_box_test, y_cls_test = load_dataset_split("test")
    
    print(f"    Train: {len(X_train)} görüntü | Val: {len(X_val)} | Test: {len(X_test)}")
    
    model = build_tensorflow_model()
    model.compile(
        optimizer=optimizers.Adam(learning_rate=5e-4),
        loss={
            "class_output": "binary_crossentropy",
            "box_output": "mse"
        },
        loss_weights={
            "class_output": 2.0,
            "box_output": 10.0
        }
    )
    
    print("[+] TensorFlow / Keras İnce Ayar Eğitimi Başlatılıyor...")
    start_time = time.time()
    history = model.fit(
        X_train,
        {"class_output": y_cls_train, "box_output": y_box_train},
        validation_data=(X_val, {"class_output": y_cls_val, "box_output": y_box_val}),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )
    training_time = time.time() - start_time
    print(f"[+] TensorFlow Eğitimi Tamamlandı. Süre: {training_time:.2f} saniye")
    
    # Modeli Kaydetme
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tf_model_path = MODELS_DIR / "pufferfish_tf_model.keras"
    model.save(tf_model_path)
    
    # Doğrulama setinde optimum eşik arama
    val_pred_cls, _ = model.predict(X_val, batch_size=BATCH_SIZE)
    best_thresh = 0.5
    best_f1 = 0.0
    for thresh in np.linspace(0.05, 0.95, 91):
        p_bin = (val_pred_cls.flatten() > thresh).astype(int)
        tp = np.sum((p_bin == 1) & (y_cls_val == 1))
        fp = np.sum((p_bin == 1) & (y_cls_val == 0))
        fn = np.sum((p_bin == 0) & (y_cls_val == 1))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)
            
    # Test Değerlendirmesi
    print(f"[+] TensorFlow Test Seti Değerlendirmesi (Eşik: {best_thresh:.3f})...")
    pred_cls, pred_box = model.predict(X_test, batch_size=BATCH_SIZE)
    
    pred_binary = (pred_cls.flatten() > best_thresh).astype(int)
    tp = np.sum((pred_binary == 1) & (y_cls_test == 1))
    fp = np.sum((pred_binary == 1) & (y_cls_test == 0))
    fn = np.sum((pred_binary == 0) & (y_cls_test == 1))
    
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1_score = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    ious = [compute_iou(pred_box[i], y_box_test[i]) for i in range(len(X_test)) if y_cls_test[i] == 1]
    mean_iou = float(np.mean(ious)) if len(ious) > 0 else 0.0
    map50_estimate = float(np.mean(np.array(ious) >= 0.5)) if len(ious) > 0 else 0.0
    
    # Gecikme Ölçümü
    dummy_input = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    for _ in range(10):
        _ = model.predict(dummy_input, verbose=0)
        
    t0 = time.time()
    n_runs = 50
    for _ in range(n_runs):
        _ = model.predict(dummy_input, verbose=0)
    latency_ms = ((time.time() - t0) / n_runs) * 1000.0
    
    model_size_mb = float(os.path.getsize(tf_model_path) / (1024 * 1024))
    
    tf_metrics = {
        "model": "TensorFlow MobileNetV2 Detector",
        "epochs": EPOCHS,
        "img_size": IMG_SIZE,
        "optimal_threshold": round(best_thresh, 3),
        "total_training_time_sec": round(training_time, 2),
        "test_metrics": {
            "accuracy": float(np.mean(pred_binary == y_cls_test)),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4),
            "mean_iou": round(mean_iou, 4),
            "mAP50_estimate": round(map50_estimate, 4)
        },
        "latency_ms_per_frame": round(latency_ms, 2),
        "model_size_mb": round(model_size_mb, 2)
    }
    
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    tf_metrics_path = BENCHMARK_DIR / "tensorflow_metrics.json"
    with open(tf_metrics_path, "w", encoding="utf-8") as f:
        json.dump(tf_metrics, f, indent=4, ensure_ascii=False)
        
    print(f"[+] TensorFlow metrikleri kaydedildi: {tf_metrics_path}")
    return tf_metrics

if __name__ == "__main__":
    train_and_eval_tensorflow()
