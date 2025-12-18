# 🐡 Underwater Pufferfish Detection System

Raspberry Pi 5 için optimize edilmiş, gerçek zamanlı balon balığı tespit sistemi. YOLO11m modeli ve ONNX Runtime kullanılarak INT8 quantization ile çalışır.

## 🎯 Proje Özeti

Bu proje, Raspberry Pi 5 donanımında verimli şekilde çalışan bir sualtı balon balığı tespit sistemi uygulamaktadır. Sistem şunları kullanır:

- **YOLO11m** - Nesne tespiti için (Medium model - hız/doğruluk dengesi)
- **ONNX Runtime** - CPU üzerinde optimize edilmiş inference
- **OpenCV** - Görüntü ön işleme ve kamera entegrasyonu
- **INT8 Quantization** - Edge deployment için optimize edilmiş model
- **Lab-Color CLAHE** - Sualtı görüntü iyileştirme

## 🚀 Özellikler

- ✅ YOLO11m ile gerçek zamanlı balon balığı tespiti
- ✅ PC'de NVIDIA CUDA ile GPU hızlandırmalı eğitim
- ✅ Raspberry Pi 5 için INT8 optimize model
- ✅ Headless çalışma modu (ekransız deployment)
- ✅ Display modunda canlı görüntüleme
- ✅ GPIO entegrasyonu (LED/alarm tetikleme - Pin 17)
- ✅ Otomatik deployment scriptleri
- ✅ Tespit anında otomatik fotoğraf kaydetme

## 📁 Proje Yapısı

```
Balik_Projesi_Antigravity/
│
├── 📁 app/                          # Runtime Uygulaması (Raspberry Pi 5)
│   ├── main_pi.py                   # Pi runtime (ekranlı mod)
│   ├── main_headless.py             # Pi runtime (headless mod)
│   └── utils/
│       └── img_processing.py        # CLAHE preprocessing
│
├── 📁 models/                       # Model Dosyaları
│   ├── yolo11m_pufferfish.pt        # Eğitilmiş PyTorch model
│   └── pufferfish_pi_int8.onnx      # Production INT8 model
│
├── 📁 training/                     # Eğitim Scriptleri (PC - CUDA)
│   ├── data_prep.py                 # Dataset hazırlama
│   ├── train_yolo.py                # YOLO11m eğitimi
│   ├── train_export_pc.py           # Eğitim + ONNX export + quantize
│   └── export_quantize.py           # INT8 quantization
│
├── 📁 scripts/                      # Deployment & Kurulum
│   ├── deploy_to_pi.bat             # Windows → Pi deployment
│   ├── install_pi.sh                # Pi kurulum scripti
│   ├── baslat.sh                    # Pi başlatma scripti
│   └── export_for_pi.py             # Pi için model export
│
├── � docs/                         # Dokümantasyon
│   ├── design_strategy.md           # Tasarım stratejisi
│   └── research/                    # Araştırma dökümanları
│
├── 📁 dataset/                      # Eğitim verileri (gitignored)
│
├── 📄 README.md                     # Bu dosya
├── 📄 requirements.txt              # Python bağımlılıkları
├── 📄 LICENSE                       # MIT Lisans
├── 📄 SECURITY.md                   # Güvenlik bilgisi
├── 📄 .env.example                  # Örnek environment
└── 📄 .gitignore                    # Git ignore kuralları
```

## 🛠️ Kurulum

### PC Kurulumu (Windows - Eğitim)

1. **Depoyu klonlayın:**
```bash
git clone https://github.com/kysrgit/Balik_Projesi.git
cd Balik_Projesi
```

2. **Sanal ortam oluşturun:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

3. **Bağımlılıkları yükleyin:**
```bash
pip install -r requirements.txt
```

4. **GPU desteği için (NVIDIA CUDA):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Raspberry Pi 5 Kurulumu

1. **Dosyaları Pi'ye aktarın:**
```bash
# Windows'ta scripts klasöründen çalıştırın
cd scripts
deploy_to_pi.bat
```

2. **Pi üzerinde kurulum yapın:**
```bash
chmod +x install_pi.sh
./install_pi.sh
```

3. **Uygulamayı başlatın:**
```bash
# Ekranlı mod
python3 app/main_pi.py

# Headless mod (arka planda çalışma)
python3 app/main_headless.py
```

## 🎓 Model Eğitimi

Eğitim sadece PC üzerinde (NVIDIA GPU ile) yapılır:

1. **Dataset hazırlama:**
```bash
python training/data_prep.py
```

2. **Model eğitimi:**
```bash
python training/train_yolo.py
```

3. **Pi için export ve quantization:**
```bash
python training/export_quantize.py
# veya
python scripts/export_for_pi.py
```

## 🖥️ Kullanım

### Raspberry Pi 5 Üzerinde

```bash
# Ekranlı mod - Canlı görüntüleme ile
python3 app/main_pi.py

# Headless mod - Tespitler diske kaydedilir
python3 app/main_headless.py
```

Tespit edilen balon balıkları `detections/` klasörüne otomatik kaydedilir.

## 📊 Performans

| Platform | Model | Precision | FPS | Latency |
|----------|-------|-----------|-----|---------|
| Raspberry Pi 5 | YOLO11m | INT8 | 5-8 | ~125ms |

## 🔧 Donanım Gereksinimleri

### PC (Eğitim)
- **OS:** Windows 10/11
- **GPU:** NVIDIA RTX 3060 veya üstü
- **RAM:** 16GB+
- **Depolama:** 20GB+ boş alan

### Raspberry Pi 5 (Deployment)
- **Model:** Raspberry Pi 5 (4GB/8GB RAM önerilir)
- **Kamera:** V4L2 uyumlu USB kamera veya Pi Camera Module
- **Depolama:** 32GB+ microSD kart
- **GPIO:** Pin 17 - LED/alarm bağlantısı (opsiyonel)

## ⚙️ Konfigürasyon

### Tespit Parametreleri

`app/main_pi.py` ve `app/main_headless.py` içinde:

```python
CONF_THRESHOLD = 0.60  # Güven eşiği (0.0 - 1.0)
MODEL_PATH = "models/pufferfish_pi_int8.onnx"  # Model dosyası
DETECTION_DIR = "detections"  # Kayıt klasörü
```

### GPIO Ayarları

LED/alarm için GPIO Pin 17 kullanılmaktadır. Tespit anında LED yanar.

## 📝 Teknik Detaylar

### Preprocessing Pipeline
1. Kamera görüntüsü alınır (640x640, YUYV format)
2. Lab color space'e çevrilir
3. CLAHE (Contrast Limited Adaptive Histogram Equalization) uygulanır
4. BGR'ye geri çevrilir
5. Model inference yapılır

### Model Bilgisi
- **Mimari:** YOLO11m (Medium)
- **Giriş Boyutu:** 640x640
- **Quantization:** INT8 (Dynamic)
- **Çıkış:** Bounding boxes + confidence scores

## 🤝 Katkıda Bulunma

Katkılarınız memnuniyetle karşılanır! Pull Request göndermekten çekinmeyin.

## 📄 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır - detaylar için LICENSE dosyasına bakın.

## 🙏 Teşekkürler

- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)
- [ONNX Runtime](https://onnxruntime.ai/)
- [OpenCV](https://opencv.org/)

## 📧 İletişim

Sorular veya destek için GitHub üzerinden issue açabilirsiniz.

---

**Not:** Bu proje sualtı balon balığı tespiti için optimize edilmiştir. Dataset ve eğitilmiş modeller boyut kısıtlamaları nedeniyle depoya dahil değildir.
