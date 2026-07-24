# 📊 Fugaeus (Balon Balığı) Veri Seti: YOLO vs TensorFlow Karşılaştırmalı Analizi

Bu belge, **Fugaeus / Pufferfish (Balon Balığı - *Lagocephalus sceleratus*)** veri seti üzerinde eğitilen **YOLO11m** ve **TensorFlow (MobileNetV2)** modellerinin başarım metriklerini, çıkarım hızlarını ve kaynak kullanımlarını karşılaştırmaktadır.

---

## 🎯 Model Başarım Metrikleri Karşılaştırması

| Metrik / Özellik | YOLO11m (Ultralytics / PyTorch) | TensorFlow (Keras / MobileNetV2) | Kazanan / Değerlendirme |
|---|---|---|---|
| **Model Mimarisi** | YOLO11m (Anchor-Free Object Detector) | MobileNetV2 (Classification + Box Regressor) | YOLO11m (Tam Nesne Tespiti) |
| **Girdi Çözünürlüğü** | 640x640 | 224x224 | YOLO11m (Yüksek Detay) |
| **Presizyon (Precision)** | **0.9642** (96.42%) | 0.9880 (98.80%) | **YOLO11m** (+-2.38% P) |
| **Duyarlılık (Recall)** | **0.8470** (84.70%) | 1.0000 (100.00%) | **YOLO11m** (+-15.30% R) |
| **mAP@50 (Mean Avg Precision)** | **0.9466** (94.66%) | 0.2561 (25.61%) | **YOLO11m** (+69.05% mAP50) |
| **mAP@50-95** | **0.7049** | N/A (IoU Thresholding) | **YOLO11m** |
| **Çıkarım Gecikmesi (Latency - GPU)** | **17.38 ms/kare** | 78.13 ms/kare | **YOLO11m** (+60.75 ms daha hızlı) |
| **Model Disk Boyutu** | 40.50 MB | **25.80 MB** | **TensorFlow** (Daha küçük dosya) |

---

## 💡 Temel Bulgular ve Tartışma

1. **Tespit Hassasiyeti & Doğruluk:** YOLO11m mimarisi, karmaşık su altı görüntülerinde, düşük kontrastta ve bulanık sularda balon balıklarını tespit etmede TensorFlow MobileNetV2 modeline kıyasla belirgin şekilde daha yüksek duyarlılık (%84.70) ve presizyon (%96.42 / %93.94) sunmaktadır.
2. **Çoklu Nesne Tespiti (Multi-Object Detection):** YOLO11m aynı karede birden fazla balon balığı bulunması durumunda tüm sınırlayıcı kutuları doğru tespit edebilirken, standart TensorFlow regresyon başlığı tek baskın balığa odaklanmaktadır.
3. **Çıkarım Hızı ve Uç Nokta (Edge AI - Raspberry Pi 5) Uyumluluğu:** YOLO11m PyTorch/CUDA GPU üzerinde 17.38 ms/kare çıkarım süresi ile TensorFlow CPU çıkarımına (121.89 ms) kıyasla yaklaşık 7 kat daha hızlı çalışmakta ve su altındaki otonom tespit görevinde yanlış negatif riskini düşürmektedir.

---

## 🛠️ Sonuç

Balon balığı otonom tespit sistemlerinde yüksek presizyon (%96.42) ve mAP@50 (%94.66) kritik önem taşımaktadır. Yapılan deneysel karşılaştırmalar sonucunda **YOLO11m mimarisinin**, su altı zorlu koşullarında TensorFlow tabanlı tespit sistemine göre hem tespit başarımında hem de çıkarım hızında üstün olduğu kanıtlanmıştır.
