# 🐡 Balon Balığı Tespit Sistemi

Raspberry Pi 5 üzerinde çalışan, gerçek zamanlı sualtı balon balığı (*Lagocephalus sceleratus*) tespit sistemi. YOLO11m modeli, Pi 5 native kamera entegrasyonu (libcamera/picamera2), GPS geotagging, SpatiaLite veritabanı, canlı web dashboard ve uluslararası veri paylaşım altyapısı (GBIF/OBIS DarwinCore) içerir.

## Sistem Mimarisi

```
app/
├── core/                    # Çekirdek modüller
│   ├── config.py            # Merkezi ayarlar
│   ├── camera.py            # Pi 5 native kamera + OpenCV fallback
│   ├── detector.py          # YOLO ONNX wrapper (INT8)
│   ├── gpio.py              # LED kontrolü (GPIO 17)
│   └── gps.py               # GPS okuyucu (pyserial + pynmea2)
├── utils/
│   └── image.py             # CLAHE ve görüntü işleme
├── dashboard/
│   ├── server.py            # Flask + Socket.IO (Eventlet) web sunucu
│   ├── stream.py            # MJPEG streaming & FrameBuffer
│   └── templates/
│       └── index.html       # Dashboard arayüzü
├── db/
│   └── spatial.py           # SpatiaLite veritabanı katmanı
├── export/                  # Veri paylaşım modülleri
│   ├── formats.py           # GeoJSON, CSV, DarwinCore Archive
│   └── webhook.py           # Webhook bildirim sistemi
└── main.py                  # Headless/GUI çalıştırıcı + CSV loglama

training/
├── data_prep.py             # Dataset hazırlama (YOLO formatı)
├── train_yolo.py            # Model eğitimi (YOLO11m, 50 epoch)
├── train_yolo_retrain.py    # YOLO11m yeniden eğitim betiği
├── train_eval_tensorflow.py # TensorFlow / Keras benchmark modeli ve eğitimi
└── export_quantize.py       # ONNX export + INT8 quantization

scripts/
├── install_pi.sh            # Raspberry Pi 5 kurulum betiği
├── baslat.sh                # Servis başlatıcı
├── gps_simulator.py         # GPS simülatörü (geliştirme amaçlı)
└── compare_yolo_tensorflow.py # YOLO vs TensorFlow karşılaştırma rapor betiği

models/
└── pufferfish_pi_int8.onnx  # INT8 quantized ONNX model

tests/                       # Pytest test altyapısı (64 test)
├── test_detector.py         # Model, inference, CLAHE testleri
├── test_logging.py          # CSV loglama testleri
├── test_spatial.py          # Veritabanı testleri
├── test_e2e_chain.py        # Uçtan uca zincir testleri
└── test_export.py           # Export & webhook testleri
```

## 📊 Fugaeus (Balon Balığı) YOLO vs TensorFlow Benchmark Karşılaştırması

Detaylı karşılaştırma analizi ve metrik raporu için: [TENSORFLOW_VS_YOLO_COMPARISON.md](file:///c:/AI/Balik_Projesi_Antigravity/TENSORFLOW_VS_YOLO_COMPARISON.md)

| Metrik / Özellik | YOLO11m (Ultralytics / PyTorch) | TensorFlow (Keras / MobileNetV2) | Kazanan |
|---|---|---|---|
| **Model Mimarisi** | YOLO11m (Anchor-Free Detector) | MobileNetV2 Regressor | **YOLO11m** |
| **Presizyon (Precision)** | **0.6821** (%68.21) | 0.0000 (%0.00) | **YOLO11m** |
| **Duyarlılık (Recall)** | **0.5252** (%52.52) | 0.0000 (%0.00) | **YOLO11m** |
| **mAP@50** | **0.5809** (%58.09) | 0.0000 (%0.00) | **YOLO11m** |
| **Çıkarım Gecikmesi (GPU)** | **17.38 ms/kare** | 121.89 ms/kare | **YOLO11m** |
| **Model Disk Boyutu** | 38.63 MB | **23.50 MB** | **TensorFlow** |

## Gereksinimler

### Donanım (Üretim)
- Raspberry Pi 5 (8GB / 16GB)
- Raspberry Pi Camera Module 3 (IMX708)
- Active Cooler
- GPS modülü (UART, /dev/ttyAMA0)
- LED (GPIO Pin 17)

### Geliştirme (PC)
- Windows / Linux / macOS
- USB webcam (kamera fallback)
- Python 3.11+

### Yazılım
- Raspberry Pi OS (Debian Trixie) veya Windows 10/11
- Python 3.11+
- picamera2 (Pi), OpenCV, Flask, Ultralytics, ONNX Runtime, pyserial, pynmea2, eventlet

## Kurulum

### Raspberry Pi 5 (Üretim)

```bash
git clone https://github.com/kysrgit/Balik_Projesi.git
cd Balik_Projesi
chmod +x scripts/install_pi.sh
./scripts/install_pi.sh
```

Kurulum betiği şunları yapar:
1. Sistem paketlerini günceller
2. Donanım kütüphanelerini OS seviyesinde kurar (picamera2, opencv, flask)
3. `--system-site-packages` ile sanal ortam oluşturur
4. Yapay zeka kütüphanelerini pip ile indirir (ultralytics, onnxruntime)

### Windows / PC (Geliştirme)

```bash
git clone https://github.com/kysrgit/Balik_Projesi.git
cd Balik_Projesi
pip install -r requirements-dev.txt
```

> **Not:** PC'de kamera otomatik olarak OpenCV VideoCapture üzerinden çalışır (picamera2 gerekmez).

### Eğitim / Quantization Ortamı

```bash
pip install -r requirements-train.txt
```

> **Not:** `training/export_quantize.py` için `onnx`, `Pillow` ve kalibrasyon verisi gerekir.

## Kullanım

### Web Dashboard (Önerilen)
```bash
# Pi'de:
source .venv_pi/bin/activate
python3 app/dashboard/server.py

# PC'de:
python app/dashboard/server.py

# Tarayıcıdan: http://<ip>:5000
```

### Headless Mod
```bash
python3 app/main.py
```

### GUI Mod (Ekranlı Ortam)
```bash
python3 app/main.py --gui
```

### GPS Simülatörü (Geliştirme)
```bash
# Windows (TCP soketi, port 9090):
python scripts/gps_simulator.py

# Linux (PTY):
python3 scripts/gps_simulator.py
```

## Dashboard Özellikleri

- **3 Görüntü Modu:** Raw, CLAHE, Detection
- **Canlı Metrikler:** FPS, confidence, CPU sıcaklığı, throttle durumu, GPS durumu
- **Alternatif Akış:** Düşük bant genişliği için WebSocket Base64 Streaming
- **Ayarlanabilir Parametreler:** Confidence eşiği, CLAHE clip limit, overlay CLAHE / overlay boxes
- **Tespit Logu:** Zaman damgalı kayıtlar ve thumbnail önizleme
- **Anlık Bildirimler:** Toast notification ile tespit uyarısı
- **Snapshot & Kayıt:** Anlık görüntü alma, Record ON/OFF kontrolü
- **GIS Haritası:** Leaflet.js + heatmap ile canlı tespit haritası
- **Veri Export:** CSV, GeoJSON, DarwinCore Archive tek tıkla indirme
- **Webhook Yönetimi:** Slack/Discord/Teams bildirim ekleme/silme
- **Opsiyonel API Token:** `DASHBOARD_API_TOKEN` ayarlıysa mutating endpoint'ler korunur
