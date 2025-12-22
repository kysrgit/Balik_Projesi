# Balon Balığı Tespit Sistemi

Raspberry Pi 5 için sualtı balon balığı tespit sistemi. YOLO modeli ve gerçek zamanlı web dashboard içerir.

## Proje Yapısı

```
app/
├── core/                 # Cekirdek moduller
│   ├── config.py         # Merkezi ayarlar
│   ├── camera.py         # Kamera (Pi + OpenCV)
│   ├── detector.py       # YOLO wrapper
│   └── gpio.py           # LED kontrolu
├── utils/
│   └── image.py          # CLAHE ve goruntu isleme
├── dashboard/
│   ├── server.py         # Flask web server
│   ├── stream.py         # MJPEG streaming
│   └── templates/        # HTML
├── main.py               # Ana calistirici

training/
├── data_prep.py          # Dataset hazirlama
├── train_yolo.py         # Model egitimi
└── export_quantize.py    # ONNX export + INT8

scripts/
├── deploy_to_pi.bat      # Pi'ye transfer
└── install_pi.sh         # Pi kurulum

models/
└── pufferfish_pi_int8.onnx  # Egitilmis model
```

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanim

### Konsol modu (headless)
```bash
python app/main.py
```

### GUI modu (pencereli)
```bash
python app/main.py --gui
```

### Web Dashboard
```bash
python app/dashboard/server.py
# http://localhost:5000
```

## Ozellikler

- YOLO11m model, INT8 quantized
- Lab-CLAHE sualti goruntu iyilestirme
- 30 FPS gercek zamanli tespit
- Web dashboard (MJPEG stream)
- GPIO LED uyarisi
- Otomatik tespit kaydi

## Donanim

- Raspberry Pi 5 (16GB)
- Camera Module 3
- Active Cooler
- LED (GPIO 17)

## Model Egitimi

1. Gorselleri `balon_baligi_fotograflari/` klasorune koy
2. `python training/data_prep.py` - dataset olustur
3. Roboflow veya LabelImg ile etiketle
4. `python training/train_yolo.py` - egit
5. `python training/export_quantize.py` - Pi icin export

## Ayarlar

`app/core/config.py` dosyasinda:
- `CONF_THRESH`: Tespit esigi (varsayilan 0.60)
- `SKIP_FRAMES`: Performans icin frame atlama
- `CLAHE_CLIP`: Kontrast ayari

---
Son guncelleme: 2024-12
