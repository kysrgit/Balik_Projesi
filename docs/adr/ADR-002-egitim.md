# ADR-002: Eğitim verisi, tarif ve mimari

**Durum:** Önerildi (E2 deneyleri sürüyor)
**Tarih:** 2026-10-19
**Faz:** 2
**Karar verenler:** Kayra (proje sahibi)

## Bağlam
E1-a ([ADR-001](ADR-001-veri.md)) mevcut modelin üç zayıflığını gösterdi:
1. Zor negatiflerde (başka balık türleri) %49 yanlış alarm. Hedef K4 ≤ %5.
2. Hareket bulanıklığında recall %88'den %46'ya düşüyor. Hedef K7: en fazla 10 puan kayıp.
3. Küçük ve kalabalık sahnelerde recall %67.

Model boyutu (n/s/m) ve hassasiyet farkları güven aralıkları içinde kaldı. Bu yüzden Faz 2'de **önce veri ve tarif**, sonra mimari ele alınır.

## Yöntem düzeltmesi
Önceki eğitim betiği (`train_v2_improved.py`) en iyi modeli **test setine bakarak** seçiyordu. Bu, test sonucunu iyimser gösterir. Yeni deneylerde:
- Model seçimi (best.pt) yalnızca **val_v2** ile yapılır. val_v2, train'den kaynak düzeyinde ayrılan %15'lik kısım (93 pozitif) ve zor negatiflerin %15'inden (52) oluşur.
- **gold_v1'e yalnızca en sonda bir kez bakılır.** gold_v1 ile aynı pHash grubunda ya da aynı iNaturalist gözleminde olan hiçbir görüntü eğitime girmez.

## Deney seti E2 (tek değişken, tek tohum ile tarama)
Sabitler: YOLO11s (COCO ön eğitimli), 640 px, `train_e2.py` içindeki tarif, 60 epoch, patience 15, tohum 0.

| Deney | Eğitim verisi | Görüntü | Sınadığı soru |
|---|---|---|---|
| E2-0 base | Mevcut train (val_v2 hariç) | 1611 | Yeni ayrımla temel çizgi |
| E2-1 hardneg | base + 296 iNat zor negatif | 1907 | Zor negatif K4'ü düzeltir mi? |
| E2-2 blur | base + 809 hareket-bulanık kopya | 2420 | Bulanıklık artırması K7'yi düzeltir mi? |
| E2-3 full | base − 52 sentetik negatif + zor negatif + bulanık | 2645 | Birleşik etki |

Not: Varyantların görüntü sayısı farklı ve epoch sayısı sabit; büyük setler daha çok güncelleme adımı görür. Bu fark sonuç yorumunda dikkate alınacak.

Komutlar:
```bash
python scripts/build_trainset.py
python training/train_e2.py --variants E2-0_base E2-1_hardneg E2-2_blur E2-3_full
```
Sonuçlar: `experiments/E2_training.json`, `experiments/E2_gold_<deney>.json`.

## Karar kuralı
1. Her deney gold_v1 ve gold_v1_blur üzerinde ADR-000 ölçütleriyle değerlendirilir.
2. Güven aralığı temel çizgiden ayrışan iyileşmeler "gerçek" kabul edilir.
3. Kazanan veri tarifi **3 tohumla** tekrarlanır (ortalama ± standart sapma).
4. Ardından aynı veriyle mimari karşılaştırması yapılır: YOLO11n/s ve YOLO26n/s. Seçim, ADR-000 K5'i (Pi'de ≥ 10 FPS) karşılayanlar arasından yapılır.

## Sonuçlar (E2, tek tohum, 2026-10-20)
gold_v1 üzerinde; IoU 0,5; köşeli parantezler bootstrap %95 güven aralığı. Kaynak: `experiments/E2_gold_*.json`. Eğitim süreleri sırasıyla 28 / 33 / 43 / 47 dakika (RTX 3060).

**Eşik 0,4** (ADR-000: recall önceliği; K1'i sağlayan en yüksek eşik aranıyor):

| Deney | Recall (K1 ≥ %80) | Precision (K3 ≥ %70) | Zor negatifte alarm (K4 ≤ %5) | Bulanıkta recall | Bulanıklık kaybı (K7 ≤ 10) | Küçük nesne (n=13) | mAP@50 |
|---|---|---|---|---|---|---|---|
| E2-0 base | %90,6 [84–96] | %64,0 ❌ | %58,9 ❌ | %49,8 | 40,8 puan ❌ | %46 | 0,858 |
| E2-1 + zor negatif | %83,4 [77–89] | %92,9 | %5,4 | %51,9 | 31,5 puan ❌ | %38 | **0,915** |
| E2-2 + bulanıklık | %90,2 [84–95] | %63,1 ❌ | %56,2 ❌ | **%83,0** | **7,2 puan** ✅ | %54 | 0,848 |
| **E2-3 tam** | **%85,5** [79–91] ✅ | **%89,7** ✅ | **%5,4** (sınırda) | **%80,4** | **5,1 puan** ✅ | %54 | 0,909 |

Eşik 0,6'da E2-3: recall %81,3, precision %94,6, alarm %3,6, bulanıkta recall %75,7.

### Yorum
1. **Zor negatif (E2-1), K4'ü %59'dan %5'e indirdi** ve mAP@50'yi 0,858'den 0,915'e çıkardı. Bedeli, aynı eşikte yaklaşık 5–8 puanlık recall kaybı. Eşik düşürülerek telafi edilebiliyor.
2. **Bulanıklık artırması (E2-2), K7'yi 41 puandan 7 puana indirdi.** Temiz görüntüde recall'ü düşürmedi.
3. **İki etki birbirine karışmıyor ve toplanıyor (E2-3).** Eşik 0,4'te K1, K3 ve K7 karşılanıyor; K4 sınırda (%5,4; 112 negatiften 6'sı).
4. Mimari değil veri değişti; farklar güven aralıklarının **dışında**. Bu, ADR-001'deki "sorun veride" bulgusunu doğruluyor.

### Bu sonuçların henüz kanıtlamadıkları (açık riskler)
- **Negatif ezberi:** Eğitimdeki zor negatifler ile gold_v1'deki negatifler aynı türlerden ve aynı kaynaktan (iNat *Siganus*/*Sparisoma*) geliyor; gözlemler ise farklı. Alarm düşüşünün bir kısmı tür ezberi olabilir. Sınamak için farklı türlerden oluşan dış bir negatif seti gerekiyor: IMEDEA Mallorca (Zenodo 7534425, CC BY 4.0, 638 MB).
- **Bulanıklık ezberi:** Eğitim bulanıklığı (çekirdek 5–21) ile test bulanıklığı (9–17) aynı üreteç ailesinden. Gerçek kamera bulanıklığına genellenmesi ayrıca doğrulanmalı.
- **Tek tohum:** Kazanan tarif (E2-3) 3 tohumla tekrarlanmalı.
- Küçük nesne örneği hâlâ 13; ölçüt güvenilir değil.

### Dış negatif kontrolü: gold_v1_ext (2026-10-20)
Ezber şüphesini sınamak için IMEDEA Labelled Mediterranean Fish Dataset'ten (Zenodo 7534425, CC BY 4.0, Mallorca, 25 sınıf, hiçbiri puffer değil) pHash ile kopyaları ayıklanmış **300 görüntü** seçildi ve donduruldu (`scripts/build_gold_ext.py` → `dataset_meta/gold_v1_ext.csv`). Görüntülerin 297'sinde balık var: *Diplodus*, *Coris*, *Chromis*, *Sparus*, *Serranus*, *Mullus* ve diğerleri. Hiçbiri eğitimde kullanılmadı ve gold_v1'deki türlerden farklı. Sonuçlar: `experiments/E2_gold_ext.json`.

| Model | Dış alarm @0,25 | @0,4 | @0,6 |
|---|---|---|---|
| Yayındaki (E1-a, YOLO11s INT8) | %50,3 | %37,0 [31–42] | %22,7 |
| E2-0 base | %53,0 | %36,7 [31–42] | %23,7 |
| E2-1 + zor negatif | %31,3 | %15,0 [11–19] | %6,0 |
| E2-2 + bulanıklık | %58,3 | %42,7 [37–48] | %26,3 |
| **E2-3 tam** | **%9,3** | **%6,0** [3–9] | **%1,7** |

**Sonuç:** Zor negatif etkisi tür ezberi değil; eğitimde hiç görülmemiş Akdeniz türlerine de genelleniyor. E2-3, E2-1'den de açıkça iyi (güven aralıkları ayrışıyor). Muhtemel sebep, sentetik negatiflerin çıkarılması ve bulanıklık artırmasının birlikte etkisi; bu tek tohumla ayrıştırılamaz.

### Çalışma eşiği (E2-3, tek tohum)
| Eşik | Recall | Precision | gold_v1 alarm | Dış alarm | Bulanıkta recall |
|---|---|---|---|---|---|
| 0,4 | %85,5 | %89,7 | %5,4 | %6,0 | %80,4 |
| **0,5** | **%82,1** ✅ | **%92,8** ✅ | **%4,5** ✅ | **%2,7** ✅ | **%78,3** (kayıp 3,8) ✅ |
| 0,6 | %81,3 | %94,6 | %3,6 | %1,7 | %75,7 |

Eşik 0,5'te ölçülebilen bütün ADR-000 ölçütleri (K1, K3, K4, K7) karşılanıyor. K2 (küçük nesne) n=13 ile hâlâ ölçülemiyor. Eşik kararı kesinleşmeden önce 3 tohumla doğrulanmalı.

### Sonraki adımlar
1. ~~Dış negatif seti (IMEDEA) ile "gold_v1-dış" kontrolü.~~ ✅ Yapıldı.
2. E2-3'ün 3 tohumla tekrarı.
3. Aynı veriyle mimari karşılaştırması: YOLO11n ve YOLO26n (Pi'de ≥ 10 FPS için nano ölçek).
4. Farklı bir bulanıklık üretecinin ve gerçek bulanık görüntülerin denenmesi.
