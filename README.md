# 🐡 Balon Balığı Tespit Sistemi

Raspberry Pi 5 üzerinde çalışan, gerçek zamanlı sualtı balon balığı (*Lagocephalus sceleratus*) tespit sistemi. YOLO11s modeli, ONNX mixed-precision INT8 kuantizasyonu, Pi 5 native kamera entegrasyonu (libcamera/picamera2), GPS geotagging, SpatiaLite veritabanı, canlı web dashboard ve uluslararası veri paylaşım altyapısı (GBIF/OBIS DarwinCore) içerir.

## Sistem Mimarisi

```
app/
├── core/                    # Çekirdek modüller
│   ├── config.py            # Merkezi ayarlar (model yolu, eşikler, CORS, CLAHE)
│   ├── camera.py            # Pi 5 native kamera + OpenCV fallback
│   ├── detector.py          # YOLO11s ONNX çıkarım motoru (XNNPACK + thread tuning)
│   ├── detection_store.py   # Tespit geçmişi / durum yönetimi (thread-safe)
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
├── main.py                  # Headless/GUI çalıştırıcı + CSV loglama
└── main_pi.py               # Raspberry Pi 5 üretim giriş noktası

training/
├── data_prep.py             # Dataset hazırlama (YOLO formatı)
├── collect_negatives.py     # Negatif (balıksız) örnek toplama
├── train_v2_improved.py     # Model eğitimi (YOLO11s, önerilen tarif)
├── train_ablation_sizes.py  # n/s/m model boyutu karşılaştırması
├── export_int8_v2.py        # .pt → ONNX FP32 + karma hassasiyetli INT8
└── eval_false_positives.py  # Negatif görüntülerde yanlış alarm ölçümü

scripts/
├── install_pi.sh            # Raspberry Pi 5 kurulum betiği
├── baslat.sh                # Uygulamayı başlatır
├── benchmark_pc.py          # .pt / ONNX / INT8 gecikme + P/R/F1 ölçümü
├── gps_simulator.py         # Geliştirme için sahte GPS
└── generate_fig5.py         # Makale Şekil 5 grafiği

benchmark_pc/                # Ölçüm sonuçları (JSON)
models/                      # Model dosyaları (git'te yok, aşağıya bakın)
data/                        # Çalışma zamanı verisi: tespit logu, görüntüler, SQLite (git'te yok)
tests/                       # Pytest test altyapısı (102 test)
```

> **Model dosyaları git'e eklenmez** (`*.pt`, `*.onnx` .gitignore'da). Uygulama `models/pufferfish_s_v2_int8.onnx` dosyasını bekler; bu dosya eğitim PC'sinden kopyalanır ya da `training/export_int8_v2.py` ile yeniden üretilir.

> **Çalışma modeli:** `models/pufferfish_s_v2_int8.onnx` (YOLO11s, karma hassasiyetli INT8 — tespit başı FP32). Model yolu `app/core/config.py` içindeki `MODEL_PATH` ile değiştirilebilir.

## 🏆 Model Performansı

### Neden YOLO11s? (model boyutu ablasyonu)
Aynı veri, 640 px, aynı hiperparametreler, 50 epoch. Kaynak: `benchmark_pc/ablation_sizes.json`.

| Model | Boyut | Test mAP@50 | Test mAP@50-95 | Test Recall |
|---|---|---|---|---|
| YOLO11m (v1, arşivde) | 40,5 MB | 70,68% | 51,38% | 56,12% |
| YOLO11n | 5,5 MB | 73,81% | 54,04% | 64,75% |
| **YOLO11s** | **19,2 MB** | **75,24%** | **56,93%** | **69,06%** |

~725 özgün eğitim fotoğrafı için YOLO11m aşırı öğreniyor; daha küçük model daha iyi genelliyor. 960 px eğitim denemesi test mAP@50'yi %63,12'ye düşürdü (`benchmark_pc/training_v2_results.json`), bu yüzden 640 px'te kalındı.

### Dağıtım formatları (YOLO11s)
Test seti (83 görüntü), PC CPU (i7-11800H). Gecikme: `scripts/benchmark_pc.py` → `benchmark_pc/results_s_v2_cpu.json`. Yanlış alarm: doğrulama setindeki 33 balıksız görüntü, conf 0,60 (`training/eval_false_positives.py`).

| Format | Boyut | Test mAP@50 | Test mAP@50-95 | CPU Gecikme | Yanlış Alarm |
|---|---|---|---|---|---|
| PyTorch (.pt) | 19,2 MB | 76,50% | 58,65% | 118,5 ms | 0 / 33 |
| ONNX FP32 | 37,9 MB | 75,92% | 57,87% | 98,3 ms | 0 / 33 |
| **ONNX INT8 (karma)** | **12,5 MB** | **75,38%** | **58,20%** | **56,9 ms** | **0 / 33** |

> mAP değerleri `ultralytics val` ile CPU'da `batch=1` kullanılarak ölçüldü; ablasyon tablosundaki (GPU, batch'li) değerlerden ~1 puan farklı olabilir. Karşılaştırma için eski YOLO11m modeli aynı 33 negatifte 1 yanlış alarm verdi.
> **Raspberry Pi 5 üzerinde cihaz-içi ölçüm henüz yapılmadı.**

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

> **Not:** INT8 dışa aktarımı (`training/export_int8_v2.py`) için `onnx` ve kalibrasyon görüntüleri (`dataset/valid/images`) gerekir.

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

## Test

```bash
# Tüm test paketini çalıştır (102 test):
pip install -r requirements-dev.txt
pytest -q
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

---

## 🔄 Yeni Bir Tür İçin Yeniden Eğitim ve Sürdürülebilirlik Rehberi

Sistem başka bir istilacı veya yerli deniz türünün tespiti için yeniden kullanılabilir. Adımlar şunlardır:

### 1. Veri Setini Hazırlama
- `dataset/` klasör yapısını koruyun (`train/valid/test` × `images/labels`).
- `dataset/data.yaml` içindeki `nc` (sınıf sayısı) ve `names` (tür isimleri) alanlarını güncelleyin.
- **Kritik Kurallar:**
  - Veri setine **hedefsiz/negatif (balıksız) sualtı görüntüleri** ekleyin (en az %10–15). Test setinde de negatif görüntü bulundurun ki yanlış alarm (false positive) oranı doğru ölçülebilsin.
  - Veri veri seti split'lerini (`train/valid/test`) **kaynak görüntü (fotoğraf) seviyesinde** yapın. Aynı fotoğraftan türetilen augment kopyalar kesinlikle farklı setlere dağılmamalıdır (data leakage riski).

### 2. Model Eğitimi
- Küçük/orta ölçekli sualtı veri setlerinde (1000 resim altı) direkt `m` veya `l` modellerine atlamayın (`n` veya `s` kullanın — overfitting önleme).
- Eğitimi başlatın:
  ```bash
  .venv_gpu/Scripts/python.exe training/train_v2_improved.py --model yolo11s.pt --imgsz 640 --batch 8 --name yeni_tur --epochs 80
  ```

### 3. ONNX ve Mixed-Precision INT8 Nicemleme
Tek komut FP32 ONNX'i ve karma hassasiyetli INT8'i üretir (`models/<stem>.pt`, `<stem>.onnx`, `<stem>_int8.onnx`):
```bash
python training/export_int8_v2.py --weights runs/detect/yeni_tur/weights/best.pt --stem yeni_tur --imgsz 640
```
Ardından `app/core/config.py` içindeki `MODEL_PATH` değerini yeni INT8 dosyasına çevirin.

- **Kritik INT8 Kuralı:** Anchor-free YOLO modellerinde tüm grafı nicemlemek tespit başını bozar ve %0 doğruluk verir. Betik, tespit başını otomatik bulup FP32'de bırakır.
- **Çözünürlük Uyuşmazlığı:** Eğitim ve çıkarım çözünürlüğü **aynı olmalıdır** (640×640). 640'ta eğitilen modeli 1280'de çalıştırmak mAP@50'yi %71,3'ten %48,8'e düşürdü.

### 4. Doğrulama
```bash
python training/eval_false_positives.py     # negatiflerde yanlış alarm
python scripts/benchmark_pc.py --device cpu --out benchmark_pc/results_yeni_tur_cpu.json
```
