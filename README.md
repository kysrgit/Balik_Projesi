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
├── review/                  # PC tespit inceleme arayüzü (python -m app.review.server)
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
├── balik-dashboard.service  # systemd servis şablonu
├── baslat.sh                # Dashboard'ı elle başlatır
├── benchmark_pc.py          # .pt / ONNX / INT8 gecikme + P/R/F1 ölçümü (PC)
├── compare_models.py        # Aday modellerin doğruluk + yanlış alarm karşılaştırması
├── benchmark_pi5.py         # Pi 5 cihaz-içi gecikme ölçümü
├── gps_simulator.py         # Geliştirme için sahte GPS
└── generate_fig5.py         # Makale Şekil 5 grafiği

benchmark_pc/                # PC ölçüm sonuçları (JSON)
benchmark_pi5/               # Pi 5 ölçüm sonuçları (JSON + özet)
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

### Model × hassasiyet karşılaştırması (Pi 5 için seçim)
Altı aday aynı protokolle ölçüldü. Doğruluk PC'de ölçüldü (`scripts/compare_models.py` → `benchmark_pc/model_compare.json`); test seti 83 görüntü, yanlış alarm 33 balıksız görüntüde conf 0,60. Hız Pi 5'te ölçüldü (`scripts/benchmark_pi5.py` → `benchmark_pi5/cmp_*.json`), 200 kare, throttling yok.

| Model | Hassasiyet | Boyut | Test mAP@50 | Test mAP@50-95 | Yanlış alarm | **Pi 5 gecikme** | Pi 5 FPS |
|---|---|---|---|---|---|---|---|
| **YOLO11s** | **INT8 (karma)** | **12,5 MB** | **75,38%** | **58,20%** | **0/33** | **173 ms** | **5,77** |
| YOLO11s | FP16 | 19,0 MB | 75,92% | 57,93% | 0/33 | 368 ms | 2,72 |
| YOLO11s | FP32 | 37,9 MB | 75,92% | 57,87% | 0/33 | 365 ms | 2,74 |
| YOLO11m | INT8 (karma) | 25,1 MB | 70,06% | 51,93% | 1/33 | 375 ms | 2,67 |
| YOLO11m | FP16 | 40,3 MB | 68,88% | 51,95% | 1/33 | 1373 ms | 0,73 |
| YOLO11m | FP32 | 80,4 MB | 68,87% | 52,04% | 1/33 | 998 ms | 1,00 |

**Karar: YOLO11s + INT8 (karma hassasiyet).**
- **FP16 Pi 5'te hız kazandırmıyor.** ONNX Runtime'ın ARM CPU tarafında bu modeldeki katmanların çoğu için FP16 çekirdeği yok; bu katmanlar araya Cast (tür dönüşümü) eklenerek FP32'de çalışıyor (ölçülen sonuç bunu doğruluyor). YOLO11s'te FP16 ile FP32 aynı hızda. YOLO11m'de FP16, FP32'den %38 daha yavaş. FP16'nın tek kazancı dosya boyutunun yarıya inmesi; bunu INT8 daha iyi yapıyor. FP16 ancak GPU veya NPU olan cihazlarda (ör. Jetson, Hailo) anlamlı olur.
- **INT8 tek gerçek hızlandırma.** Pi 5'in Cortex-A76 çekirdeklerindeki tamsayı nokta çarpım komutlarını kullanıyor. YOLO11s'te INT8, FP32'den 2,1 kat hızlı ve doğruluk kaybı yarım puanın altında (mAP@50-95 hatta biraz daha yüksek). Tespit başı FP32'de bırakıldığı için kutu konumları bozulmuyor.
- **YOLO11m bu veri setinde hem daha yavaş hem daha kötü.** Test mAP@50'de 5-7 puan geride ve tek yanlış alarmı veren model. YOLO11m INT8 bile YOLO11s FP32 kadar yavaş. Sebep: yaklaşık 725 özgün eğitim fotoğrafı, YOLO11m'nin 20 milyon parametresi için az; model aşırı öğreniyor (bkz. üstteki ablasyon). YOLO11m ancak veri seti birkaç kat büyürse yeniden denenmeli.
- 16 GB RAM bu kararı değiştirmiyor. Hepsi 100 MB'ın altında olduğu için bellek darboğaz değil; darboğaz 4 çekirdekli CPU'nun hesaplama hızı.

### Ön işleme: CLAHE ve görüntü oranı
Aynı model (YOLO11s INT8) ve eşik 0,60 ile, inceleme arayüzünde (`app/review`) ölçüldü:

| | Test F1 | Valid F1 | Boş sahnede alarm |
|---|---|---|---|
| **CLAHE kapalı** | **74,6%** | **89,7%** | **0/33** |
| CLAHE açık | 68,2% | 86,5% | 2/33 |

Bu yüzden tespitten önce CLAHE varsayılan olarak **kapalı** (`DETECT_USE_CLAHE = False`). Önceden dashboard bunu her karede zorla açıyordu. Ayrıca uygulama kameradan gelen 4:3 kareyi önceden 640×640'a **geriyordu**; artık eğitimdeki gibi en-boy oranı korunarak letterbox yapılıyor.

Eşik taraması (YOLO11s INT8, valid): 0,5 → F1 %87,0; **0,6 → F1 %89,7, boş sahnede 0 alarm**; 0,7 → F1 %90,2; 0,8 → F1 %88,9. Varsayılan eşik 0,60 olarak bırakıldı; dashboard'dan değiştirilebilir.

### Dağıtım formatları (YOLO11s)
Test seti (83 görüntü), PC CPU (i7-11800H). Gecikme: `scripts/benchmark_pc.py` → `benchmark_pc/results_s_v2_cpu.json`. Yanlış alarm: doğrulama setindeki 33 balıksız görüntü, conf 0,60 (`training/eval_false_positives.py`).

| Format | Boyut | Test mAP@50 | Test mAP@50-95 | CPU Gecikme | Yanlış Alarm |
|---|---|---|---|---|---|
| PyTorch (.pt) | 19,2 MB | 76,50% | 58,65% | 118,5 ms | 0 / 33 |
| ONNX FP32 | 37,9 MB | 75,92% | 57,87% | 98,3 ms | 0 / 33 |
| **ONNX INT8 (karma)** | **12,5 MB** | **75,38%** | **58,20%** | **56,9 ms** | **0 / 33** |

> mAP değerleri `ultralytics val` (8.3.x) ile CPU'da `batch=1` kullanılarak ölçüldü. Precision/recall sürümden bağımsızdır; mAP hesabı ultralytics sürümüne göre ~0,7 puan oynayabilir (Pi'deki 8.4.x ile INT8: mAP@50 %74,52, mAP@50-95 %57,08). Karşılaştırma için eski YOLO11m modeli aynı 33 negatifte 1 yanlış alarm verdi.

### Raspberry Pi 5 üzerinde (cihaz-içi ölçüm)
Pi 5 (16 GB, aktif soğutucu), Raspberry Pi OS 64-bit (Debian 13), ONNX Runtime 1.31 CPUExecutionProvider, 4 çekirdek. Test görüntüleri, 20 ısınma + 200 kare, ön işleme + çıkarım + NMS dahil, CLAHE kapalı. Ölçüm boyunca throttling yok (en yüksek 66 °C). Betik: `scripts/benchmark_pi5.py`, sonuçlar: `benchmark_pi5/`.

| Model | Ortalama gecikme | p95 | FPS |
|---|---|---|---|
| **ONNX INT8 (karma)** | **171,1 ms** | 171,7 ms | **5,85** |
| ONNX FP32 | 362,1 ms | 364,0 ms | 2,76 |

INT8, Pi 5'te FP32'ye göre **2,1× hızlı**. Dashboard'da kamera ve tespit ayrı döngülerde çalışır. Tespit döngüsü her seferinde en son kareyi alır, eski kareleri atar. Bu yüzden görüntü akışı tespit hızından bağımsız olarak akıcı kalır.

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

Raspberry Pi OS 64-bit (Debian 13 "trixie") üzerinde test edildi.

```bash
git clone https://github.com/kysrgit/Balik_Projesi.git
cd Balik_Projesi
./scripts/install_pi.sh
```

Model dosyaları git'te olmadığı için INT8 modeli PC'den kopyalayın (PC'de proje kökünde):

```bash
scp models/pufferfish_s_v2_int8.onnx <kullanici>@<pi-adi>.local:~/Balik_Projesi/models/
```

Kurulum betiği şunları yapar (tekrar çalıştırılabilir):
1. OS paketlerini kurar: `python3-picamera2`, `rpicam-apps`, SpatiaLite (picamera2 pip ile kurulamaz)
2. `--system-site-packages` ile `.venv_pi` sanal ortamını oluşturur ve `requirements.txt`'i kurar
3. Dashboard'ı açılışta başlayan `balik-dashboard` systemd servisi olarak kurar (`--no-service` ile atlanır)

Servis yönetimi: `sudo systemctl restart balik-dashboard`, loglar: `journalctl -u balik-dashboard -f`. Kamera bağlı değilse dashboard yine açılır ve kamera hatasını gösterir.

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
# Pi'de (servis kurulduysa zaten çalışıyor):
./scripts/baslat.sh

# PC'de:
python app/dashboard/server.py

# Tarayıcıdan: http://<ip>:5000
```

### Tespit İnceleme Arayüzü (PC) 🔍
Modelin balon balığını doğru bulup bulmadığını gözle kontrol etmek için:
```bash
python -m app.review.server
# Tarayıcıdan: http://localhost:5050
```
- **Etiketli veri setinde (test/valid/train):** Etiket ile tahminler üst üste çizilir. Yeşil kutu doğru tespit, kırmızı kutu yanlış alarm, turuncu kesikli kutu kaçırılan balık anlamına gelir. Precision, recall, F1 ve boş sahnede alarm sayısı canlı hesaplanır.
- **Güven eşiği kaydırıcısı:** Eşiği değiştirince sonuçlar anında yeniden hesaplanır.
- **Filtreler:** "Hatalı olanlar", "Kaçırılan var", "Yanlış alarm var", "Boş sahneler".
- **Büyük görünüm:** Bir karta tıklayınca açılır; ← → ile gezilir.
- **Kendi dosyaların:** Görüntü veya video yükleyebilirsin. Videodan saniyede yaklaşık 1 kare örneklenir (en fazla 120 kare).
- **Model ve CLAHE seçimi:** `models/` ve `models/candidates/` altındaki herhangi bir model seçilebilir, CLAHE açılıp kapatılabilir.

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
