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
│   ├── server.py            # Flask + Socket.IO web sunucu
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
└── export_quantize.py       # ONNX export + INT8 quantization

scripts/
├── install_pi.sh            # Raspberry Pi 5 kurulum betiği
├── baslat.sh                # Servis başlatıcı
└── gps_simulator.py         # GPS simülatörü (geliştirme amaçlı)

models/
└── pufferfish_pi_int8.onnx  # INT8 quantized ONNX model

tests/                       # Pytest test altyapısı (64 test)
├── test_detector.py         # Model, inference, CLAHE testleri
├── test_logging.py          # CSV loglama testleri
├── test_spatial.py          # Veritabanı testleri
├── test_e2e_chain.py        # Uçtan uca zincir testleri
└── test_export.py           # Export & webhook testleri
```

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
- picamera2 (Pi), OpenCV, Flask, Ultralytics, ONNX Runtime, pyserial, pynmea2

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
pip install -r requirements.txt
```

> **Not:** PC'de kamera otomatik olarak OpenCV VideoCapture üzerinden çalışır (picamera2 gerekmez).

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
- **Canlı Metrikler:** FPS, confidence, CPU sıcaklığı, throttle durumu
- **Ayarlanabilir Parametreler:** Confidence eşiği, CLAHE clip limit (canlı slider)
- **Tespit Logu:** Zaman damgalı kayıtlar ve thumbnail önizleme
- **Anlık Bildirimler:** Toast notification ile tespit uyarısı
- **Snapshot & Kayıt:** Anlık görüntü alma, Record ON/OFF kontrolü
- **GIS Haritası:** Leaflet.js + heatmap ile canlı tespit haritası
- **Veri Export:** CSV, GeoJSON, DarwinCore Archive tek tıkla indirme
- **Webhook Yönetimi:** Slack/Discord/Teams bildirim ekleme/silme

## Veri Paylaşım Katmanı

Uluslararası araştırma kuruluşlarıyla veri paylaşımı için 4 kanal:

| Endpoint | Format | Hedef Kullanım |
|---|---|---|
| `GET /api/export/csv` | CSV | Araştırmacılar, Excel, R/Python analiz |
| `GET /api/export/geojson` | GeoJSON FeatureCollection | QGIS, Leaflet, ArcGIS, MapBox |
| `GET /api/export/darwincore` | ZIP (DwC-A) | **GBIF, OBIS** — uluslararası biyoçeşitlilik ağları |
| `POST /api/webhooks` | JSON | Slack, Discord, Teams, özel API bildirimleri |

### DarwinCore Archive İçeriği
GBIF ve OBIS'e doğrudan yüklenebilir standart format:
- `occurrence.csv` — Tespit kayıtları (DwC standart sütunları)
- `meta.xml` — Arşiv tanımlayıcı
- `eml.xml` — Ekolojik metadata (tür, yöntem, coğrafi kapsam)

### Webhook Bildirimleri
```bash
# Webhook ekle:
curl -X POST http://localhost:5000/api/webhooks \
  -H "Content-Type: application/json" \
  -d '{"name": "slack", "url": "https://hooks.slack.com/services/..."}'

# Listele:
curl http://localhost:5000/api/webhooks

# Sil:
curl -X DELETE http://localhost:5000/api/webhooks \
  -H "Content-Type: application/json" \
  -d '{"name": "slack"}'
```

## Test Altyapısı

64 otomatik test ile tüm bileşenler doğrulanmıştır:

```bash
python -m pytest tests/ -v
```

| Test Dosyası | Kapsam | Test Sayısı |
|---|---|---|
| `test_detector.py` | ONNX model, inference, CLAHE, draw_boxes | 11 |
| `test_logging.py` | CSV başlık, yazma, append, boş tespit | 5 |
| `test_spatial.py` | DB insert, confidence/region sorgusu, şema | 6 |
| `test_e2e_chain.py` | Tespit→CSV+thumb, API, queue, recording gate | 13 |
| `test_export.py` | GeoJSON, CSV, DarwinCore, webhook, endpoint'ler | 29 |
| **Toplam** | | **64** |

## Teknik Detaylar

| Parametre | Değer |
|---|---|
| Model | YOLO11m, INT8 quantized ONNX |
| Giriş Çözünürlüğü | 640x640 |
| Kamera Çözünürlüğü | 640x480 @ 30 FPS |
| Kamera Arayüzü | libcamera (Pi) / OpenCV VideoCapture (PC) |
| Inference | ONNX Runtime (XNNPACK Pi / CPU PC) |
| Ön İşleme | Lab renk uzayında CLAHE |
| GPS | pyserial + pynmea2 (GPGGA/GPRMC) |
| Veritabanı | SpatiaLite (spatial index + MakePoint) |
| Streaming | MJPEG over HTTP |
| İletişim | Flask-SocketIO (WebSocket, threading mode) |
| Veri Standartı | DarwinCore (GBIF/OBIS uyumlu) |

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
| `GPS_PORT` | /dev/ttyAMA0 | GPS UART portu |
| `GPS_BAUDRATE` | 9600 | GPS baud rate |
| `DASHBOARD_SAVE_INTERVAL` | 1.0 | Max 1 tespit kaydı/saniye |
| `MAX_MAP_POINTS` | 5000 | Haritada max nokta sayısı |

## API Referansı

| Endpoint | Method | Açıklama |
|---|---|---|
| `/` | GET | Dashboard ana sayfası |
| `/video/<stream_type>` | GET | MJPEG stream (raw/clahe/detection) |
| `/api/config` | GET/POST | Ayar okuma/güncelleme |
| `/api/record` | POST | Kayıt aç/kapat toggle |
| `/api/snapshot` | POST | Anlık görüntü kaydet |
| `/api/export/csv` | GET | CSV indirme |
| `/api/export/geojson` | GET | GeoJSON indirme |
| `/api/export/darwincore` | GET | DarwinCore ZIP indirme |
| `/api/webhooks` | GET/POST/DELETE | Webhook yönetimi |

## Lisans

MIT License

---
Son güncelleme: 2026-03
