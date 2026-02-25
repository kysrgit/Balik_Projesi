# 🐡 Balon Balığı Tespit Sistemi

Raspberry Pi 5 üzerinde çalışan, gerçek zamanlı sualtı balon balığı tespit sistemi. YOLO11m modeli, Pi 5 native kamera entegrasyonu (libcamera/picamera2) ve canlı web dashboard içerir.

## Sistem Mimarisi

```
app/
├── core/                    # Çekirdek modüller
│   ├── config.py            # Merkezi ayarlar
│   ├── camera.py            # Pi 5 native kamera (picamera2)
│   ├── detector.py          # YOLO ONNX wrapper
│   └── gpio.py              # LED kontrolü (GPIO 17)
├── utils/
│   └── image.py             # CLAHE ve görüntü işleme
├── dashboard/
│   ├── server.py            # Flask + Socket.IO web sunucu
│   ├── stream.py            # MJPEG streaming & FrameBuffer
│   └── templates/
│       └── index.html       # Dashboard arayüzü
└── main.py                  # Headless/GUI çalıştırıcı

training/
├── data_prep.py             # Dataset hazırlama (YOLO formatı)
├── train_yolo.py            # Model eğitimi (YOLO11m, 50 epoch)
└── export_quantize.py       # ONNX export + INT8 quantization

scripts/
└── install_pi.sh            # Raspberry Pi 5 kurulum betiği

models/
└── pufferfish_pi_int8.onnx  # INT8 quantized ONNX model
```

## Gereksinimler

### Donanım
- Raspberry Pi 5 (8GB / 16GB)
- Raspberry Pi Camera Module 3 (IMX708)
- Active Cooler
- LED (GPIO Pin 17)

### Yazılım
- Raspberry Pi OS (Debian Trixie)
- Python 3.11+
- picamera2, OpenCV, Flask, Ultralytics, ONNX Runtime

## Kurulum (Raspberry Pi 5)

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

## Kullanım

### Web Dashboard (Önerilen)
```bash
source .venv_pi/bin/activate
python3 app/dashboard/server.py
# Tarayıcıdan: http://<pi_ip>:5000
```

### Headless Mod
```bash
python3 app/main.py
```

### GUI Mod (Ekranlı Ortam)
```bash
python3 app/main.py --gui
```

## Dashboard Özellikleri

- **3 Görüntü Modu:** Raw, CLAHE, Detection
- **Canlı Metrikler:** FPS, confidence, CPU sıcaklığı, throttle durumu
- **Ayarlanabilir Parametreler:** Confidence eşiği, CLAHE clip limit
- **Tespit Logu:** Zaman damgalı kayıtlar ve thumbnail önizleme
- **Anlık Bildirimler:** Toast notification ile tespit uyarısı
- **Snapshot & Kayıt:** Anlık görüntü alma

## Teknik Detaylar

| Parametre | Değer |
|---|---|
| Model | YOLO11m, INT8 quantized ONNX |
| Giriş Çözünürlüğü | 640x640 |
| Kamera Çözünürlüğü | 640x480 @ 30 FPS |
| Kamera Arayüzü | libcamera (picamera2) |
| Inference | ONNX Runtime (CPUExecutionProvider) |
| Ön İşleme | Lab renk uzayında CLAHE |
| Streaming | MJPEG over HTTP |
| İletişim | Flask-SocketIO (WebSocket) |

## Model Eğitimi

1. Görselleri `balon_baligi_fotograflari/` klasörüne koy
2. `python training/data_prep.py` — Train/val bölümlemesi
3. LabelImg veya Roboflow ile YOLO formatında etiketle
4. `python training/train_yolo.py` — 50 epoch eğitim (GPU önerilir)
5. `python training/export_quantize.py` — INT8 ONNX export

## Ayarlar

Tüm ayarlar `app/core/config.py` dosyasında:

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `CONF_THRESH` | 0.60 | Tespit güven eşiği |
| `SKIP_FRAMES` | 5 | N frame'de bir tespit (performans) |
| `CLAHE_CLIP` | 3.0 | Kontrast iyileştirme seviyesi |
| `TARGET_FPS` | 30 | Hedef kamera FPS |
| `DASHBOARD_PORT` | 5000 | Web sunucu portu |

## Lisans

MIT License

---
Son güncelleme: 2026-02
