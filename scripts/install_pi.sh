#!/bin/bash
echo "====================================="
echo "🐡 Pufferfish Pi 5 Kurulum Araci 🐡"
echo "====================================="

echo "[1/4] Sistem guncelleniyor..."
sudo apt-get update && sudo apt-get upgrade -y

echo "[2/4] Gerekli sistem paketleri OS seviyesinde kuruluyor (Cok Onemli!)..."
# Pi 5'in ana kameraya (libcamera) tam erisimi olmasi icin kütüphaneleri apt ile kurmaliyiz
sudo apt-get install -y python3-venv python3-opencv python3-picamera2 python3-flask python3-flask-socketio python3-numpy python3-eventlet libgl1-mesa-glx htop

echo "[3/4] Virtual Environment (Sanal Ortam) kuruluyor..."
# Eski bozuk ortami sil
if [ -d ".venv_pi" ]; then
    echo "Eski sanal ortam siliniyor..."
    rm -rf .venv_pi
fi

# Yeni ortam (OS kutuphanelerine, yani kameraya erisim izni ile)
python3 -m venv --system-site-packages .venv_pi

echo "[4/4] Yapay Zeka (AI) kutuphaneleri indiriliyor..."
source .venv_pi/bin/activate

# Diger UI/Kamera bilesenleri yukarida "apt" ile kuruldu. 
# Pip sadece "ultralytics, torch ve ONNX" kuracak
pip install ultralytics onnxruntime pyyaml pillow

echo "====================================="
echo "✅ KUSURSUZ (GOLDEN) KURULUM TAMAMLANDI!"
echo "Projeyi Pi uzerinde baslatmak icin:"
echo "1. source .venv_pi/bin/activate"
echo "2. python3 app/dashboard/server.py"
echo "====================================="
