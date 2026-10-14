#!/bin/bash
# Raspberry Pi 5 kurulum betigi (Raspberry Pi OS 64-bit, Debian 13 "trixie")
# Kullanim:  ./scripts/install_pi.sh            -> paketler + sanal ortam + servis
#            ./scripts/install_pi.sh --no-service
# Tekrar tekrar calistirilabilir; var olan sanal ortami silmez.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
INSTALL_SERVICE=1
[ "${1:-}" = "--no-service" ] && INSTALL_SERVICE=0

echo "====================================="
echo "🐡 Balon Baligi - Pi 5 Kurulumu"
echo "====================================="

echo "[1/4] Sistem paketleri kuruluyor (sudo)..."
# picamera2/libcamera pip ile kurulamaz, OS paketinden gelmeli.
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev python3-picamera2 rpicam-apps \
    libsqlite3-mod-spatialite libgl1

echo "[2/4] Sanal ortam (.venv_pi, sistem paketlerine erisimli)..."
if [ ! -x .venv_pi/bin/python ]; then
    python3 -m venv --system-site-packages .venv_pi
fi

echo "[3/4] Python paketleri kuruluyor..."
.venv_pi/bin/pip install --upgrade pip
.venv_pi/bin/pip install -r requirements.txt

MODEL="models/pufferfish_s_v2_int8.onnx"
if [ ! -f "$MODEL" ]; then
    echo "⚠️  $MODEL bulunamadi. Model dosyalari git'te yok; PC'den kopyalayin:"
    echo "    scp models/pufferfish_s_v2_int8.onnx $(whoami)@$(hostname).local:$ROOT_DIR/models/"
fi

if [ "$INSTALL_SERVICE" = "1" ]; then
    echo "[4/4] systemd servisi kuruluyor (balik-dashboard)..."
    sed -e "s|__USER__|$(whoami)|g" -e "s|__ROOT__|$ROOT_DIR|g" \
        scripts/balik-dashboard.service | sudo tee /etc/systemd/system/balik-dashboard.service > /dev/null
    sudo systemctl daemon-reload
    sudo systemctl enable --now balik-dashboard
    sudo systemctl --no-pager status balik-dashboard | head -5 || true
else
    echo "[4/4] Servis atlandi (--no-service)."
fi

echo "====================================="
echo "✅ Kurulum tamamlandi!"
echo "Dashboard:  http://$(hostname).local:5000"
echo "Elle baslatmak icin:  ./scripts/baslat.sh"
echo "Servis loglari:       journalctl -u balik-dashboard -f"
echo "====================================="
