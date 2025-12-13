# 🐡 Underwater Pufferfish Detection System

A real-time pufferfish detection system optimized for Raspberry Pi 5, utilizing YOLO11 object detection with ONNX Runtime inference.

## 🎯 Project Overview

This project implements an underwater pufferfish detection system designed to run efficiently on Raspberry Pi 5 hardware. The system uses:

- **YOLO11n/YOLO11m** models for object detection
- **ONNX Runtime** for optimized CPU inference
- **OpenCV** for image preprocessing and camera integration
- **INT8/FP16 quantization** for edge deployment

## 🚀 Features

- ✅ Real-time pufferfish detection with YOLO11
- ✅ GPU-accelerated training and inference on PC (NVIDIA CUDA)
- ✅ Optimized INT8/FP16 models for Raspberry Pi 5
- ✅ Live monitoring with visual feedback
- ✅ Headless operation mode for deployment
- ✅ GPIO integration for LED/alarm triggers
- ✅ Automated deployment scripts

## 📁 Project Structure

```
Balik_Projesi_Antigravity/
├── app/                        # Runtime application
│   ├── live_monitor.py        # Live detection with GUI (PC)
│   ├── main_pi.py             # Raspberry Pi runtime with display
│   ├── main_headless.py       # Headless mode for Pi
│   └── ...
├── training/                   # Training scripts
│   ├── train_yolo.py          # Model training
│   ├── data_prep.py           # Dataset preparation
│   └── ...
├── dataset/                    # Training dataset (gitignored)
├── requirements.txt           # Python dependencies
├── deploy_to_pi.bat          # Deployment automation
├── install_pi.sh             # Pi installation script
└── README.md                  # This file
```

## 🛠️ Installation

### PC Setup (Windows - Training & Development)

1. **Clone the repository:**
```bash
git clone https://github.com/kysrgit/Balik_Projesi.git
cd Balik_Projesi
```

2. **Create virtual environment:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **For GPU acceleration (NVIDIA):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install onnxruntime-gpu
```

### Raspberry Pi 5 Setup

1. **Transfer files to Pi:**
```bash
# Use the automated deployment script
deploy_to_pi.bat
```

2. **Install on Pi:**
```bash
chmod +x install_pi.sh
./install_pi.sh
```

3. **Run the application:**
```bash
# With display
python app/main_pi.py

# Headless mode
python app/main_headless.py
```

## 🎓 Training

1. **Prepare your dataset:**
```bash
python training/data_prep.py
```

2. **Train the model:**
```bash
python training/train_yolo.py
```

3. **Export for deployment:**
```bash
# For Raspberry Pi (INT8)
python export_for_pi.py

# For PC (FP16)
python export_fp16.py
```

## 🖥️ Usage

### Live Monitoring (PC)
```bash
python app/live_monitor.py
```

### Raspberry Pi Deployment
```bash
# GUI mode
python app/main_pi.py

# Headless mode (saves detections to disk)
python app/main_headless.py
```

## 📊 Performance

| Platform | Model | Precision | FPS | Latency |
|----------|-------|-----------|-----|---------|
| PC (RTX 3060) | YOLO11m | FP16 | 60+ | ~16ms |
| Raspberry Pi 5 | YOLO11n | INT8 | 10-15 | ~66ms |
| Raspberry Pi 5 | YOLO11m | INT8 | 5-8 | ~125ms |

## 🔧 Hardware Requirements

### PC (Training & Development)
- **OS:** Windows 10/11
- **GPU:** NVIDIA RTX 3060 or better (recommended)
- **RAM:** 16GB+
- **Storage:** 20GB+ free space

### Raspberry Pi 5 (Deployment)
- **Model:** Raspberry Pi 5 (4GB/8GB RAM)
- **Camera:** Compatible with libcamera/V4L2
- **Storage:** 32GB+ microSD card
- **Optional:** GPIO-connected LED/alarm

## 📝 Configuration

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Edit `.env` with your settings:
- Roboflow API key (if using)
- Camera settings
- Model paths
- GPIO pin configurations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)
- [ONNX Runtime](https://onnxruntime.ai/)
- [OpenCV](https://opencv.org/)

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Note:** This project is optimized for underwater pufferfish detection. Dataset and trained models are not included in the repository due to size constraints.
