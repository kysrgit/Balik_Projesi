# 🗺️ Yol Haritası — Su Altı Balon Balığı Tespit Sistemi

> **İlke:** Hiçbir seçim (veri, model, hassasiyet, motor, donanım, dil, ön işleme) varsayımla yapılmaz.
> Her faz bir **soruyla** başlar, **aynı protokolle ölçülen** adaylarla devam eder ve sonuç bir
> **karar kaydına (ADR)** yazılır. Bir faz kapanmadan sonrakine geçilmez. Geri dönülmesi gerekirse
> eski ADR "Superseded" (yerine yenisi geldi) olarak işaretlenir, silinmez.
>
> Hazırlanış: 2026-10-15. Kullanılan çerçeveler: *system-design* (gereksinim → tasarım → ödünleşim),
> *architecture* (ADR), *testing-strategy* (test piramidi).

## Durum özeti

| Faz | Soru | Durum |
|---|---|---|
| 0 | Sistem tam olarak neyi, nerede, hangi hızla yapmalı? | ✅ [ADR-000](adr/ADR-000-gereksinimler.md) |
| 1 | Elimizdeki veri bu soruyu ölçmeye yeterli mi? Değilse ne eksik? | ✅ [ADR-001](adr/ADR-001-veri.md): gold_v1 (327) donduruldu; E1-a: recall %88 ✅, zor negatifte alarm %49 ❌, bulanıkta recall %46 ❌ |
| 2 | Hangi mimari ve hangi eğitim tarifi en iyi genelliyor? | ⏳ [ADR-002](adr/ADR-002-egitim.md): E2 veri deneyleri eğitimde |
| 3 | Eğitilen model Pi'de hangi format, hassasiyet ve motorla en iyi çalışıyor? | ⬜ (kısmi kanıt var) |
| 4 | Pi 5 CPU yeterli mi, yoksa hızlandırıcı (AI HAT+) gerekli mi? | ⏭️ Atlandı: yalnızca Pi 5 CPU (ADR-000) |
| 5 | Hangi ön işleme su altında gerçekten fayda sağlıyor? | ⬜ (CLAHE ölçüldü) |
| 6 | Uygulama yazılımı (dil, mimari) darboğaz mı? | ⬜ |
| 7 | Sistem uçtan uca ve uzun süre güvenilir mi? | ⬜ |
| 8 | Gerçek sahada (havuz → deniz) ne kadar iyi çalışıyor? | ⬜ |
| 9 | Makale, model kartı, başka türe aktarım | ⬜ |

Durum işaretleri: ⬜ başlamadı · ⏳ sürüyor · ✅ karar verildi (ADR bağlantısıyla) · ⏭️ atlandı

**Faz 0 yanıtları:** hareketli platform (ROV/tekne) · recall öncelikli · yalnızca Pi 5 CPU · yalnızca açık kaynak veri. Hedef: Pi'de ≥ 10 FPS, recall ≥ %80 (ayrıntı: ADR-000).

---

## Ortak deney protokolü (her faz için geçerli)

1. **Sabit değerlendirme seti.** Faz 1'de dondurulan "altın test seti" dışında hiçbir sette karar verilmez. Test setine bakarak ayar yapılmaz; ayarlar doğrulama (valid) setinde yapılır.
2. **Aynı koşullar.** Karşılaştırılan adaylar aynı veri, aynı çözünürlük, aynı eşik ve aynı ölçüm betiğiyle değerlendirilir. Her deney commit hash'i ile kaydedilir.
3. **İstatistik.** Test seti küçük olduğu için her doğruluk metriği **bootstrap %95 güven aralığıyla** raporlanır. İki aday arasındaki fark, güven aralıkları ayrışmıyorsa "fark yok" kabul edilir. Eğitim deneyleri **en az 3 farklı tohum (seed)** ile tekrarlanır; ortalama ± standart sapma verilir.
4. **Gecikme her zaman Pi üzerinde ölçülür.** Uçtan uca ölçülür: ön işleme + çıkarım + NMS. Dashboard duraklatılır, 20 ısınma + 200 kare koşulur, p50 ve p95 raporlanır, sıcaklık ve throttling kaydedilir. PC'de ölçülen gecikme hiçbir kararda kullanılmaz.
5. **Kayıt.** Her deney `experiments/EXP-NNN/` altında saklanır: komut, yapılandırma, sonuç JSON'u, commit. Her karar `docs/adr/ADR-NNN-*.md` dosyasına yazılır (şablon: `docs/adr/0000-template.md`).

### Ana metrikler

| Metrik | Neden önemli | Nerede ölçülür |
|---|---|---|
| **Recall @ çalışma eşiği** | Kaçırılan balık = kaçırılan gözlem | Altın test seti |
| **Yanlış alarm / saat** ve boş sahnede alarm oranı | Sahada güveni en çok bozan şey | Negatif görüntüler + balıksız video |
| **F1 @ çalışma eşiği**, mAP@50, mAP@50-95 | Genel doğruluk ve literatürle karşılaştırma | Altın test seti |
| **Küçük nesne recall'ı** (kareye göre alanı %1'in altında) | Bilinen en büyük zayıflık | Altın test setinin alt kümesi |
| **Pi uçtan uca p95 gecikme**, FPS | Gerçek zamanlılık | Pi 5 |
| Sıcaklık, throttling, (varsa) güç | Sahada kesintisiz çalışma | Pi 5, 1 saatlik test |

---

## Faz 0 — Gereksinimler ve başarı ölçütleri

**Soru:** Sistem hangi senaryoda çalışacak ve "yeterince iyi" ne demek?
Bu cevap, sonraki her fazın karar kuralını belirler. Örneğin 2 FPS yeten sabit bir istasyonla 15 FPS gereken hareketli bir ROV farklı model seçimine götürür.

**Netleşmesi gerekenler (kullanıcıdan):**
- **Kullanım senaryosu:** Sabit gözlem istasyonu mu, dalgıç kamerası mı, ROV/insansız tekne mi, yoksa önceden kaydedilmiş videoların sonradan analizi mi?
- **Hedef hız:** Saniyede en az kaç tespit gerekli? Balık 0,3–0,8 m/s yüzer; kadrajda ortalama ne kadar kalır?
- **Hata dengesi:** Kaçırılan bir balık mı daha kötü, yoksa boş bir sahnede verilen yanlış alarm mı?
- **Donanım bütçesi:** AI HAT+ alınabilir mi? Yaklaşık maliyetler: 13 TOPS sürüm ~70 $, 26 TOPS sürüm ~110 $.
- **Kamera ve kurulum:** Pi Camera Module 3 mü, su geçirmez USB kamera mı? Kayıt derinliği, ışık koşulları, enerji kaynağı (pil süresi)?

**Çıktı:** ADR-000 — sayısal kabul ölçütleri. Örnek taslak (onaylanacak):
recall ≥ %80, boş sahnede alarm ≤ %2, Pi'de en az 5 FPS, 1 saat kesintisiz çalışmada throttling yok.

---

## Faz 1 — Veri

**Soru:** Mevcut veri, sistemin *su altında* balon balığını bulup bulamadığını ölçmeye yeterli mi?

**Mevcut durum** (2026-10-15 envanteri):

| | Dosya | Özgün kaynak | Pozitif | Negatif | Örnek | Medyan kutu alanı | Alanı %1'in altında olan |
|---|---|---|---|---|---|---|---|
| train | 1879 | 725 (her biri 3 kez çoğaltılmış) | 1728 | 151 | 2217 | %11,9 | %8,8 |
| valid | 197 | 197 | 164 | 33 | 183 | %12,4 | %3,3 |
| test | 83 | 83 | 82 | **1** | 139 | %5,9 | **%14,4** |

**Bilinen sorunlar:**
- **Ortam karışık.** Test setinde çok sayıda kara/güverte fotoğrafı (yakalanmış balık) var. Proje ise su altı tespiti. Şu anki metrikler hedef ortamı tam ölçmüyor.
- **Test setinde neredeyse hiç negatif yok** (yalnızca 1 görüntü). Yanlış alarm test setinde ölçülemiyor.
- **Benzer türler yok.** Başka balıklar (özellikle diğer Lagocephalus türleri, *Sphoeroides*, *Siganus*) "negatif" olarak bulunmuyor. Model "balık var" ile "balon balığı var" ayrımını öğrenmemiş olabilir.
- **Sızıntı kontrolü yalnızca dosya adına göre yapıldı.** Görsel benzerliğe göre (pHash) yapılmadı.
- **Video yok.** Gerçek kullanım video akışı; hareket bulanıklığı ve bulanık su ölçülmedi.
- Etiketler poligon formatında. Kutuya dönüşüm otomatik yapılıyor, sorun değil, ama belgelenmeli.

**Yapılacaklar ve deneyler:**
1. **Veri denetimi (otomatik + elle):**
   - Her görüntüye ortam etiketi ver: su altı / su üstü / kara.
   - Görsel kopyaları (pHash) bul.
   - Hatalı ve eksik etiketleri inceleme arayüzüyle işaretle.
2. **Altın test seti oluştur ve dondur:**
   - Yalnızca su altı görüntüler; kaynak (dalış veya video) düzeyinde ayrılmış.
   - **En az %30 negatif:** boş sahne + başka balık türleri.
   - Eğitimde kullanılmamış birkaç video klip.
   - Hedef: en az 300 görüntü ve en az 10 video klip. Bu set bir kez oluşturulur ve bir daha değişmez.
3. **Veri genişletme:**
   - Saha videolarından kare çıkarma.
   - Açık veri setlerinden (ör. DeepFish, Akdeniz balık setleri) negatif ve benzer tür örnekleri.
   - iNaturalist ve GBIF'teki *L. sceleratus* su altı fotoğrafları (lisans kontrolüyle).
4. **Deney E1-a:** Mevcut modeli altın test setinde ölç. Bu, gerçek başlangıç noktası (baseline) olur.
5. **Deney E1-b:** Kara/güverte görüntüleri eğitimden çıkarıldığında su altı performansı değişiyor mu?

**Karar kuralı:** Altın test seti hazır olmadan Faz 2'ye geçilmez.
**Çıktı:** ADR-001 (veri sürümü, split kuralları), bir veri kartı (`docs/DATA_CARD.md`) ve dondurulmuş `dataset_v2`.

---

## Faz 2 — Model mimarisi ve eğitim tarifi

**Soru:** Bu veri boyutunda ve su altı koşullarında hangi mimari ve hangi tarif en iyi genelliyor?

**Şu ana kadarki kanıt** (eski test seti, tek tohum):
- YOLO11s > YOLO11n > YOLO11m. YOLO11m, 725 özgün fotoğrafta aşırı öğreniyor.
- 960 px eğitim test mAP@50'yi %75'ten %63'e düşürdü.
- Bu sonuçlar **yeni altın test setinde ve 3 tohumla yeniden doğrulanacak.**

**Adaylar:**

| Aday | Neden |
|---|---|
| YOLO11n, YOLO11s | Mevcut temel çizgi |
| **YOLO26n, YOLO26s** | NMS'siz, kenar cihaz odaklı, küçük nesne iyileştirmeleri; Pi 5'te YOLO11n'e göre ~%15 daha hızlı (Ultralytics) |
| YOLO11m / YOLO26m | Yalnızca Faz 4'te hızlandırıcı seçilirse ve veri büyürse |
| RT-DETR (hafif sürüm) | Yalnızca hızlandırıcı varsa; transformer tabanlı alternatif |

**Tarif deneyleri** (en iyi 1–2 mimari üzerinde, tek değişken değiştirerek):
- Giriş çözünürlüğü: 512 / 640 / 768. Eğitim ve çıkarım aynı çözünürlükte olmalı.
- Artırma (augmentation): mevcut tarif, su altı renk ve bulanıklık artırması, mosaic/mixup açık-kapalı.
- Ön eğitim: COCO ağırlıkları ile genel su altı balık setinde önceden eğitilmiş ağırlıklar.
- Negatif ve benzer tür oranı: %0 / %15 / %30.
- Küçük nesne için kırpma artırması; tiling (SAHI) Faz 5'te değerlendirilecek.

**Karar kuralı:** Altın test setinde Faz 0 ölçütlerini karşılayan adaylar arasından doğruluk ile Pi gecikmesi arasındaki en iyi dengeyi (Pareto) sağlayan aday seçilir. Fark güven aralığı içindeyse daha küçük ve hızlı model seçilir.
**Çıktı:** ADR-002 (mimari ve tarif), `training/` altında tek komutla tekrarlanabilir eğitim.

---

## Faz 3 — Dışa aktarım, hassasiyet ve çıkarım motoru (Pi 5 CPU)

**Soru:** Seçilen model Pi 5 işlemcisinde hangi format ve hassasiyetle en hızlı çalışıyor, ve bunu doğruluğunu koruyarak mı yapıyor?

**Şu ana kadarki kanıt** (Pi 5, YOLO11s, 640 px, uçtan uca):
- ONNX Runtime: FP32 365 ms, **FP16 368 ms (kazanç yok)**, INT8 karma 173 ms.
- **NCNN FP16: 162 ms** (s) ve **70 ms** (n). Pi'nin donanım FP16 aritmetiğini (`asimdhp`) kullanıyor.
- YOLO11m FP16 ONNX'te FP32'den daha yavaş (1373 ms ve 998 ms).
- spinning açık/kapalı farkı %2. İşlemci 2,4 GHz'te, throttling yok.

**Adaylar:** NCNN (FP32 / FP16 / INT8), ONNX Runtime (FP32 / INT8 karma), OpenVINO (ARM), MNN, TFLite (XNNPACK).

**Deneyler:** Her aday için Pi'de uçtan uca gecikme ölçülür ve **altın test setinde doğruluk** ölçülür. INT8 için kalibrasyon seti, kalibrasyon yöntemi ve tespit başını dışarıda bırakma seçenekleri denenir.

**Karar kuralı:** Doğruluk kaybı FP32'ye göre güven aralığı içinde kalan en hızlı seçenek seçilir.
**Çıktı:** ADR-003 ve `scripts/benchmark_pi5.py`'nin tüm formatları destekler hâle getirilmesi.

---

## Faz 4 — Donanım hızlandırıcısı

**Soru:** Pi 5 CPU, Faz 0'daki FPS hedefini karşılıyor mu? Karşılamıyorsa veya daha büyük bir model gerekiyorsa hangi hızlandırıcı?

**Adaylar:**

| Seçenek | Not |
|---|---|
| Yalnızca Pi 5 CPU | Ek maliyet yok. NCNN ile YOLO11n ~14 FPS |
| Raspberry Pi AI HAT+ 13 TOPS (Hailo-8L) | YOLOv8s için kaynaklarda ~60–80 FPS |
| Raspberry Pi AI HAT+ 26 TOPS (Hailo-8) | YOLOv8m ~50–60 FPS; daha büyük model kullanımına izin verir |
| AI HAT+ 2 (Hailo-10H, 40 TOPS) | Daha çok üretken yapay zekâ odaklı; görüntü işleme hızı Hailo-8 ile benzer |
| Raspberry Pi AI Camera (IMX500) | Model kameranın içinde çalışır; küçük modellerle sınırlı |

**Dikkat:**
- Hailo için model, Hailo Dataflow Compiler ile (x86 Linux'ta) INT8'e derlenir. Doğruluk kaybı ayrıca ölçülmeli.
- Kaynaklardaki FPS'ler çoğunlukla toplu (batch) veya üretici ölçümü. **Kendi modelimizle, kendi Pi'mizde ölçülmeden karar verilmez.**

**Deney:**
- Önce CPU'daki en iyi sonuç (Faz 3) Faz 0 hedefiyle karşılaştırılır.
- Yetmiyorsa veya doğruluk büyük modelle belirgin artıyorsa (Faz 2'de ölçülür), hızlandırıcı satın alınıp aynı protokolle ölçülür.

**Çıktı:** ADR-004.

---

## Faz 5 — Görüntü ön işleme ve küçük nesne stratejisi

**Soru:** Su altı görüntü iyileştirmesi tespiti gerçekten iyileştiriyor mu?

**Şu ana kadarki kanıt:** CLAHE zararlı çıktı. YOLO11s INT8'de test F1 %74,6'dan %68,2'ye düştü ve boş sahnede alarm 0'dan 2'ye çıktı. Bu yüzden varsayılan olarak kapalı.

**Adaylar:**
- Beyaz dengesi / gri-dünya renk düzeltmesi.
- Hafif buğu giderme (dehazing).
- Gamma düzeltmesi.
- Küçük nesneler için **tiling (SAHI)** veya ilgi bölgesi (ROI) yakınlaştırma. Bu, gecikmeyi artırır; maliyeti ayrıca ölçülür.
- **Zamansal doğrulama:** "M karenin N'inde görüldüyse alarm ver" kuralı ve takip (ByteTrack). Yanlış alarmları azaltmak ve aynı balığı tekrar saymamak için.
- **NMS IoU eşiği:** Varsayılan 0,7. İnceleme arayüzünde aynı balığa iki kutu çizildiği görüldü (fotograf_504: iki kutu arasında IoU 0,695, bastırılmadı). 0,5–0,6 denenmeli; NMS'siz YOLO26 bu sorunu yapısal olarak çözebilir (Faz 2).

**Deney:** Her yöntem altın test setinde ve video kliplerde ölçülür, ayrıca eklediği gecikme kaydedilir.
**Çıktı:** ADR-005.

---

## Faz 6 — Uygulama yazılımı, dil ve mimari

**Soru:** Model dışındaki kod (kamera yakalama, ön işleme, akış, kayıt) kare süresinin ne kadarını tüketiyor? Python yeterli mi?

**Yöntem:** Pi'de dashboard çalışırken her aşamanın süresini profille: yakalama, renk dönüşümü, çıkarım, NMS, kutu çizimi, JPEG kodlama, ağ.

**Karar kuralı:** Model dışı süre kare süresinin %20'sinden azsa **Python'da kal** (bakım kolaylığı). Fazlaysa önce darboğazı düzelt; C++'a geçiş yalnızca ölçülen bir kazanç varsa düşünülür.

**Bilinen teknik borç:**
- `eventlet` kullanımdan kalkıyor (deprecated); threading veya asyncio tabanlı bir sunucuya geçiş gerekebilir.
- Kamera ve tespit döngüsü tasarımı gözden geçirilmeli.
- Takip (tracking) ve sayma eklenmeli.

**Çıktı:** ADR-006.

---

## Faz 7 — Sistem testi (test piramidi)

| Katman | Kapsam | Durum |
|---|---|---|
| Birim testleri | Yardımcı fonksiyonlar, dışa aktarım biçimleri, GPS ayrıştırma, güvenlik | ✅ 103 test |
| Entegrasyon testleri | Kamera → tespit → kayıt → dashboard zinciri (kamera taklidiyle) | Kısmi |
| **Model regresyon testi** | Altın test setinde recall, F1 ve alarm oranı belirlenen eşiğin altına düşerse CI başarısız olur | ⬜ |
| **Pi performans testi** | Her sürümde `benchmark_pi5.py`; p95 gecikme bütçeyi aşarsa uyarı | ⬜ |
| **Uzun süre (soak) testi** | 8 saat kesintisiz çalışma: bellek sızıntısı, sıcaklık, throttling, disk dolması | ⬜ |
| **Video tekrar testi** | Kayıtlı saha videoları kamera yerine verilir; tespitler beklenen sonuçla karşılaştırılır | ⬜ |

**Çıktı:** CI'da model regresyon testi ve bir test planı.

---

## Faz 8 — Saha doğrulaması

**Soru:** Laboratuvar sonuçları gerçek suda tutuyor mu?

1. **Kontrollü ortam:** Akvaryum veya havuzda, ekranda veya basılı balon balığı görüntüleriyle hızlı kontrol.
2. **Sığ deniz:** Kayıtlı video ve GPS ile. Örnek kareler elle etiketlenir; recall ve yanlış alarm/saat hesaplanır.
3. Sahada başarısız olunan örnekler (zor negatifler ve kaçırılan balıklar) veri setine geri beslenir. Faz 1'e dönülerek **veri döngüsü** kurulur.

**Çıktı:** Saha raporu, ADR-008.

---

## Faz 9 — Yayın ve sürdürülebilirlik

- Makale, yalnızca ADR'lerle desteklenen sayılarla güncellenir. Pi üzerindeki ölçümler artık var.
- Model kartı: eğitim verisi, sınırlılıklar, beklenen performans.
- "Başka bir tür için yeniden eğitim" rehberi: Faz 1–3 komutlarının sırayla çalıştırılması.

---

## Faz bağımlılıkları

```
Faz 0 (gereksinim) ──► Faz 1 (veri + altın test seti)
                              │
                              ▼
                       Faz 2 (mimari + tarif) ◄──────────┐
                              │                          │
                              ▼                          │ büyük model
                       Faz 3 (format/hassasiyet/motor)   │ gerekirse
                              │                          │
                              ▼                          │
                       Faz 4 (hızlandırıcı?) ────────────┘
                              │
                     ┌────────┴────────┐
                     ▼                 ▼
              Faz 5 (ön işleme)   Faz 6 (yazılım)
                     └────────┬────────┘
                              ▼
                       Faz 7 (sistem testi) ──► Faz 8 (saha) ──► Faz 9 (yayın)
                                                    │
                                                    └──► Faz 1'e geri besleme
```

## Gözden geçirme noktaları

- **Veri 3 katına çıkarsa:** Faz 2'de daha büyük modeller (s → m) yeniden denenir.
- **Hızlandırıcı alınırsa:** Faz 2 ve 3 hızlandırıcı için tekrarlanır.
- **Sahada yanlış alarm yüksekse:** Önce Faz 1'e (zor negatifler), sonra Faz 5'e (zamansal doğrulama) dönülür.
