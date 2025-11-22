# Design Strategy: Underwater Pufferfish Detection System (2025 Upgrade)

## 1. Hardware Constraints & Optimization Goal
**Target Hardware:** Raspberry Pi 5 (16GB RAM), Camera Module 3.
**Constraint:** No NPU (Hailo/Coral). **CPU-Only Inference.**
**Goal:** Achieve real-time performance (10-15 FPS) with high accuracy.

## 2. Model Selection: YOLO11n (Nano)
We have selected **YOLO11n** over the previous YOLOv8s for the following reasons:
- **Parameter Efficiency:** YOLO11n is designed to be significantly lighter than v8s, making it ideal for CPU-bound devices.
- **Latency:** The 'Nano' architecture minimizes floating-point operations (FLOPs), directly translating to faster inference times on the ARM Cortex-A76 CPU of the RPi 5.
- **Accuracy:** Despite being smaller, YOLO11 architectures incorporate improved feature extraction layers that maintain competitive accuracy, especially for distinct objects like pufferfish.

## 3. Preprocessing: Lab-Color Space CLAHE
Underwater imagery suffers from poor contrast and color absorption (blue/green dominance). Standard CLAHE on RGB images can distort colors.
**Our Approach:**
1.  **Convert RGB to CIE Lab:** Separate Lightness (L) from Color (a, b).
2.  **Apply CLAHE to 'L' Channel:** Enhance local contrast without touching color information.
3.  **Merge & Convert back to RGB:** Result is a contrast-enhanced image with natural color preservation.

This ensures the model receives clear structural features without color artifacts that could confuse classification.

## 4. Deployment Pipeline
- **Training:** Fine-tuning on Roboflow dataset.
- **Export:** ONNX (Open Neural Network Exchange) for broad compatibility.
- **Quantization:** **INT8 Dynamic Quantization**. This reduces model size by ~4x and speeds up inference on CPU by using integer arithmetic instead of floating point, which is critical for the RPi 5.
