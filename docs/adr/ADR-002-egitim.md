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

## Sonuçlar
*(E2 deneyleri bitince doldurulacak)*
