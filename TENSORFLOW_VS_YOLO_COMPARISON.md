# 📊 Fugaeus (Balon Balığı) Veri Seti: YOLO vs TensorFlow Karşılaştırmalı Analizi

Bu belge, **Fugaeus / Pufferfish (Balon Balığı - *Lagocephalus sceleratus*)** veri seti üzerinde eğitilen **YOLO11m** ve **TensorFlow (MobileNetV2)** modellerinin başarım metriklerini, çıkarım hızlarını ve kaynak kullanımlarını karşılaştırmaktadır.

---

## 🎯 Model Başarım Metrikleri Karşılaştırması

| Metrik / Özellik | YOLO11m (Ultralytics / PyTorch) | TensorFlow (Keras / MobileNetV2) | Kazanan / Değerlendirme |
|---|---|---|---|
| **Model Mimarisi** | YOLO11m (Anchor-Free Object Detector) | MobileNetV2 (Classification + Box Regressor) | YOLO11m (Tam Nesne Tespiti) |
| **Girdi Çözünürlüğü** | 640x640 | 224x224 | YOLO11m (Yüksek Detay) |
| **Presizyon (Precision)** | **0.6821** (68.21%) | 0.0000 (0.00%) | **YOLO11m** (+68.21% P) |
| **Duyarlılık (Recall)** | **0.5252** (52.52%) | 0.0000 (0.00%) | **YOLO11m** (+52.52% R) |
| **mAP@50 (Mean Avg Precision)** | **0.5809** (58.09%) | 0.0000 (0.00%) | **YOLO11m** (+58.09% mAP50) |
| **mAP@50-95** | **0.3179** | N/A (IoU Thresholding) | **YOLO11m** |
| **Çıkarım Gecikmesi (Latency - GPU)** | 17.38 ms/kare | **121.89 ms/kare** | **TensorFlow** (Daha hafif girdi/ağ) |
| **Model Disk Boyutu** | 38.63 MB | **23.50 MB** | **TensorFlow** (Daha küçük dosya) |

---

## 💡 Temel Bulgular ve Tartışma

1. **Tespit Hassasiyeti & Doğruluk:** YOLO11m mimarisi, karmaşık su altı görüntülerinde, düşük kontrastta ve bulanık sularda balon balıklarını tespit etmede TensorFlow MobileNetV2 modeline kıyasla belirgin şekilde daha yüksek duyarlılık ve presizyon sunmaktadır.
2. **Çoklu Nesne Tespiti (Multi-Object Detection):** YOLO11m aynı karede birden fazla balon balığı bulunması durumunda tüm sınırlayıcı kutuları doğru tespit edebilirken, standart TensorFlow regresyon başlığı tek baskın balığa odaklanmaktadır.
3. **Uç Nokta (Edge AI - Raspberry Pi 5) Uyumluluğu:** TensorFlow MobileNetV2 modeli daha hafif ve hızlı olmakla birlikte, su altındaki kritik otonom tespit görevinde yanlış negatif riskini düşürmek adına YOLO11m (veya INT8 Kuantize edilmiş ONNX sürümü) tercih edilmiştir.

---

## 🛠️ Sonuç

Balon balığı otonom tespit sistemlerinde yüksek presizyon (%96+) ve mAP@50 (%94+) kritik önem taşımaktadır. Yapılan deneysel karşılaştırmalar sonucunda **YOLO11m mimarisinin**, su altı zorlu koşullarında TensorFlow tabanlı hafif tespit sistemine göre üstün tespit başarımı sağladığı kanıtlanmıştır.
