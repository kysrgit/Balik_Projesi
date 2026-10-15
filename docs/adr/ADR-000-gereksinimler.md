# ADR-000: Gereksinimler ve sayısal kabul ölçütleri

**Durum:** Önerildi (sayısal eşikler kullanıcı onayı bekliyor)
**Tarih:** 2026-10-15
**Faz:** 0
**Karar verenler:** Kayra (proje sahibi)

## Bağlam
Sonraki her fazın karar kuralı bu ölçütlere dayanır. Kullanıcı yanıtları (2026-10-15):

| Soru | Yanıt |
|---|---|
| Kullanım senaryosu | **Hareketli platform (ROV / tekne)** |
| Daha kötü hata | **Balığı kaçırmak** (recall öncelikli) |
| Hızlandırıcı (AI HAT+) | **Hayır, yalnızca Raspberry Pi 5 (16 GB) CPU** |
| Yeni veri | **Yalnızca açık kaynaklar** (saha videosu yok) |

## Karar — kabul ölçütleri (önerilen)

| # | Ölçüt | Hedef | Gerekçe |
|---|---|---|---|
| K1 | **Recall** (altın test seti, çalışma eşiğinde) | **≥ %80** | Öncelik kaçırmamak |
| K2 | Küçük nesne recall'ı (kareye göre alanı %1'in altında) | ≥ %50 | Hareketli platformda balık çoğu zaman uzakta ve küçük |
| K3 | Precision | ≥ %70 | Recall öncelikli; ama kayıtların yarıdan fazlası çöp olmamalı |
| K4 | Boş sahnede alarm (negatif görüntüler) | ≤ %5 | |
| K5 | **Pi 5 uçtan uca tespit hızı** | **≥ 10 FPS (p95 ≤ 100 ms)** | Kamera hareket ettiği için balık kadrajda kısa kalır; ≥ 10 FPS ile her balık birden çok karede görülür ve zamansal doğrulama (Faz 5) mümkün olur |
| K6 | 1 saat kesintisiz çalışma | Throttling yok, bellek artışı yok | ROV görev süresi |
| K7 | Hareket bulanıklığına dayanıklılık | Bulanıklaştırılmış test kopyasında recall kaybı ≤ 10 puan | ROV ve tekne titreşimi |

## Sonuçları
- **Faz 4 (hızlandırıcı) atlanır.** Tüm modeller Pi 5 CPU'da çalışmak zorunda.
- **K5, aday havuzunu daraltır.** Mevcut ölçümlere göre (Pi 5, 640 px) yalnızca nano boyutlu modeller 10 FPS'e yaklaşıyor: YOLO11n NCNN 70 ms. YOLO11s NCNN 162 ms ile K5'i karşılamıyor. Faz 2'de ana adaylar YOLO11n ve YOLO26n olacak. s boyutu ancak daha düşük çözünürlükte (ör. 480 veya 512 px) K5'i karşılarsa değerlendirilir.
- **Recall önceliği:** Çalışma eşiği valid setinde K1'i sağlayan en yüksek eşik olarak seçilir; bu muhtemelen 0,60'ın altında olacak. Yanlış alarmı azaltmak için zamansal doğrulama (Faz 5) kullanılır.
- **Veri yalnızca açık kaynaklardan:** Faz 1'de video yerine fotoğraf ağırlıklı bir altın test seti kurulur. Hareket koşulları sentetik bulanıklık ve titreşim artırmasıyla taklit edilir (K7).

## Ne zaman yeniden değerlendirilmeli
- Saha videosu elde edilirse K2 ve K7 gerçek verilerle yeniden ölçülür.
- Hızlandırıcı bütçesi çıkarsa Faz 4 açılır, K5'in getirdiği model boyutu sınırı kalkar.

## Yapılacaklar
1. [ ] Kullanıcı K1–K7 eşiklerini onaylasın veya değiştirsin.
2. [ ] Faz 1: veri denetimi ve altın test setinin oluşturulması.
