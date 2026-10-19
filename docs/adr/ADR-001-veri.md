# ADR-001: Tespit kapsamı, veri kaynakları ve altın test seti (gold_v1)

**Durum:** Kabul edildi
**Tarih:** 2026-10-15 (kapsam) · 2026-10-19 (gold_v1 ve E1-a)
**Faz:** 1
**Karar verenler:** Kayra (proje sahibi)

## Bağlam
Veri denetimi ([DATA_AUDIT.md](../DATA_AUDIT.md)) şu sorunları gösterdi:
1. Test setinin yaklaşık yarısı su dışında çekilmiş; neredeyse hiç negatif görüntü yok; 1 görüntü train'den sızmış.
2. Etiket sınıfı genel olarak `pufferfish`; veri setinde *L. sceleratus* dışındaki puffer türleri de bulunuyor.
3. Valid setindeki 33 "negatifin" 13'ü programla üretilmiş sentetik görüntü, 6'sı düşük kaliteli arka plan kırpıntısı. Bu yüzden önceden raporlanan "boş sahnede 0/33 yanlış alarm" sonucu gerçek su altı sahnelerini ölçmüyordu.

ADR-000'a göre saha verisi yok; yalnızca açık kaynak kullanılacak.

## Karar
1. **Kapsam: tüm istilacı (Lesepsiyen) puffer türleri tek sınıf** (`pufferfish`). Pozitif türler: *Lagocephalus sceleratus*, *L. suezensis*, *L. guentheri*, *Torquigener flavimaculosus*.
2. **Zor negatifler:** Aynı habitatta yaşayan, puffer olmayan türler: *Siganus luridus*, *S. rivulatus*, *Sparisoma cretense*.
3. **Açık kaynak:** iNaturalist; yalnızca araştırma kalitesinde ve açık lisanslı fotoğraflar. Lisans, fotoğrafçı ve gözlem bağlantısı `dataset_meta/inat_photos.csv` dosyasında. İndirilen 1.744 fotoğrafın **452'si zaten Roboflow veri setinde** çıktı (pHash ile tespit edildi); bunlar altın sete alınmadı.
4. **Altın test seti `gold_v1` dondurulmuştur** (`dataset_meta/gold_v1.csv`: her dosyanın SHA-256 değeri, kökeni ve lisansı). Bundan sonraki her karar bu sette ölçülür. Değişiklik gerekirse `gold_v2` oluşturulur; `gold_v1` değiştirilmez.

## gold_v1 içeriği

| Köken | Görüntü | Not |
|---|---|---|
| dataset/test (suda) | 39 | Kullanıcının kutu incelemesi + düzeltmeleri |
| dataset/valid (suda) | 96 | 84 pozitif + 12 gerçek boş sahne |
| iNaturalist pozitif | 92 | 70 su altı, 22 su üstünden; hiç eğitimde yok |
| iNaturalist zor negatif | 100 | *Siganus* / *Sparisoma*, suda |
| **Toplam** | **327** | 215 pozitif (235 balık), 112 negatif |

Ayrıca `dataset_gold_v1_blur/`: aynı görüntülerin sabit tohumlu hareket bulanıklığı uygulanmış kopyası (ADR-000 K7).

### Nasıl etiketlendi (izlenebilirlik)
- **Test seti kutuları:** Kullanıcı tarafından incelendi (79/83 fotoğraf; `label_fixes.json`).
- **Ortam etiketleri ve valid kutuları:** Claude tarafından görsel kontrolle verildi; kayıtlarda `not = "claude: ..."` ile işaretli. Emin olunmayan 37 fotoğraf ayrıca "emin degil" diye işaretli.
- **iNaturalist kutuları:** Model önyargısını önlemek için **iki bağımsız dedektörün** önerileri birleştirildi: YOLO ve hiç bizim veriyle eğitilmemiş OWLv2 (açık sözlük, "fish"). Doğru kutular Claude tarafından seçildi; kayıtlarda `by = "claude"`. Belirsiz 24 fotoğraf hariç tutuldu.
- Kullanıcı herhangi bir kaydı `/denetim`, `/etiket` ve `/` sayfalarından kontrol edip düzeltebilir.

## E1-a: Mevcut modelin gerçek başlangıç noktası
`scripts/eval_gold.py` → `experiments/E1a_gold_baseline.json`. IoU 0,5, bootstrap %95 güven aralığı (2000 tekrar), eşik 0,60.

| Model | Recall (K1 ≥ %80) | Precision (K3 ≥ %70) | Zor negatifte alarm (K4 ≤ %5) | Küçük nesne recall (K2 ≥ %50) | Bulanıkta recall (K7) | mAP@50 |
|---|---|---|---|---|---|---|
| **YOLO11s INT8 (yayında)** | **%87,7** [81,4–93,4] ✅ | %73,3 ✅ | **%49,1** ❌ | %46 (n=13) ❌ | %46,0 (−42 puan) ❌ | 0,862 |
| YOLO11s FP32 | %87,7 | %71,8 | %49,1 | %38 | %49,4 | 0,868 |
| YOLO11n INT8 | %84,3 | %71,0 | %45,5 | %38 | %46,8 | 0,833 |
| YOLO11n FP32 | %85,1 | %68,3 | %50,9 | %46 | %49,8 | 0,827 |
| YOLO11m FP32 | %84,7 | %71,6 | %44,6 | %31 | %49,4 | 0,820 |

Kökene göre (YOLO11s INT8):

| Köken | Recall | Precision | Negatifte alarm |
|---|---|---|---|
| iNat pozitif (yeni) | %94,6 | %97,8 | — |
| dataset/valid | %92,2 | %97,6 | 0/12 |
| **dataset/test** (küçük, kalabalık) | **%67,3** | %100 | — |
| **iNat zor negatif** | — | — | **55/100** |

### Bulgular
1. **Model "balon balığı"nı değil "balığı" öğrenmiş.** Başka türden her iki balıktan biri puffer sanılıyor (K4: %49). Eğitim verisinde başka balık türü yok. Bu, sahadaki en büyük hata kaynağı olur.
2. **Hareket bulanıklığına dayanıksız** (K7: recall %88 → %46). ROV/tekne senaryosu için kritik.
3. **Küçük ve kalabalık sahnelerde zayıf** (dataset/test recall %67). Küçük nesne örneği yalnızca 13; bu ölçüt için gold_v2'de daha çok örnek gerekir.
4. **Model boyutu (n/s/m) ve hassasiyet (INT8/FP32) farkları güven aralıkları içinde.** Bu aşamada model değiştirmek sorunu çözmez; **sorun veride.**

### Sınırlılıklar
- Valid'den gelen 96 görüntü, mevcut modelin eğitiminde erken durdurma ve model seçimi için kullanıldı; bu kısım hafif iyimser olabilir.
- Gerçek boş sahne sayısı az (12); hedef ≥ 40 idi.
- Etiketlerin bir kısmı insan yerine Claude tarafından verildi (yukarıda izlenebilir şekilde işaretli).

## Sonuçları (Faz 2'ye girdi)
Faz 2'nin önceliği mimari değil **eğitim verisi ve tarifi**:
1. Eğitime zor negatif ekle: iNaturalist'te kullanılmamış ~330 *Siganus*/*Sparisoma* fotoğrafı, ayrıca başka yaygın Akdeniz türleri.
2. Hareket bulanıklığı artırması ekle (K7).
3. Sentetik negatifleri eğitimden çıkar (gerçek sahneyi temsil etmiyor).
4. Su dışı fotoğrafların eğitimdeki etkisini ölç (E1-b).
5. Ancak bunlardan sonra mimari karşılaştırması (YOLO11n/s, YOLO26n/s) yapılmalı.

## Yapılacaklar
1. [x] iNaturalist indirmesi (`scripts/fetch_inat.py`)
2. [x] Ortam etiketlerinin doğrulanması
3. [x] Yeni pozitifler için kutu etiketleme (YOLO + OWLv2 bağımsız önerileri)
4. [x] `gold_v1` oluşturulup donduruldu (`scripts/build_gold.py`)
5. [x] E1-a ölçüldü (`scripts/eval_gold.py`)
6. [ ] Kullanıcı: Claude'un "emin degil" işaretlediği kayıtlara göz atması (isteğe bağlı)
