# Veri Denetimi Raporu (Faz 1)

**Tarih:** 2026-10-15 · **Betik:** `scripts/data_audit.py` · **Ham çıktı:** `dataset_meta/audit.csv`, `dataset_meta/audit_summary.json`
**Doğrulama arayüzü:** `python -m app.review.server` → http://localhost:5050/denetim

## 1. Envanter

| | Dosya | Özgün kaynak | Pozitif | Negatif | Medyan kutu alanı | Alanı %1'in altında |
|---|---|---|---|---|---|---|
| train | 1879 | 725 (×~2,6 Roboflow artırma kopyası) | 1728 | 151 | %11,9 | %8,8 |
| valid | 197 | 197 | 164 | 33 | %12,4 | %3,3 |
| test | 83 | 83 | 82 | **1** | %5,9 | **%14,4** |
| ham klasör | 827 | — | — | — | — | — |

- Tüm görüntüler 640×640 (Roboflow tarafından yeniden boyutlandırılmış). Etiketler poligon formatında (1974 dosya), 185 dosya boş (negatif).
- Ham klasördeki 827 fotoğrafın **tamamı** veri setinde var. Veri setindeki **181 kaynak** ham klasörde yok, Roboflow tarafında eklenmiş.

## 2. Kopya ve sızıntı (pHash, Hamming ≤ 6)

| Bulgu | Etki |
|---|---|
| `test:fotograf_71` ≈ `train:fotograf_81` (neredeyse aynı görüntü) | Test setinde 1 sızıntı. Altın setten çıkarılacak. |
| `train:fotograf_486` ≈ `train:fotograf_535` | Train içinde kopya; zararsız ama gereksiz. |
| Dosya adına göre sızıntı | Yok. |

## 3. Ortam dağılımı (CLIP önerisi, **henüz doğrulanmadı**)

| | Su altı | Su üstünden (suda, yukarıdan) | Su dışında (kara/el/güverte) |
|---|---|---|---|
| train (1879 dosya) | 604 | 708 | 567 |
| valid (197) | 66 | 83 | 48 |
| test (83) | **22** | 29 | **32** |

CLIP önerisinin 24 rastgele test görüntüsünde gözle kontrolü:
- Kara ve su altı sınıfları büyük ölçüde doğru.
- "Su üstünden" sınıfı genellikle sığ suda yukarıdan çekilmiş balıklar. Bu, **tekneden bakan kamera görünümüne** karşılık geliyor; hedef ortamın bir parçası.
- Hatalar var. Örneğin su üzerinde elde tutulan bir balık "su üstü" olarak işaretlenmiş. Bu yüzden **son etiket insan doğrulamasıyla** verilir.

**Sonuç:** Mevcut test setinin yaklaşık **%40'ı su dışı fotoğraf**, yani hedef ortamı (ROV ve tekne) temsil etmiyor. Şimdiye kadar raporlanan test metrikleri, sahadaki performansın güvenilir bir ölçüsü değil.

## 4. Tür kapsamı (açık soru)

Etiket sınıfı genel olarak `pufferfish`. Görsel kontrolde *L. sceleratus*'a benzemeyen benekli puffer türleri de görüldü (muhtemelen *Torquigener flavimaculosus* veya başka *Lagocephalus* türleri). Hedef yalnızca *L. sceleratus* ise bunlar **negatif** olmalı; tüm istilacı puffer türleri hedefse etiketleri doğru. Bu, veri setinin anlamını değiştiren bir karar (ADR-001'de yazılacak).

## 5. Açık kaynak veri havuzu (iNaturalist, 2026-10-15)

| Takson | Fotoğraflı gözlem | Araştırma kalitesi + açık lisans | Kullanım |
|---|---|---|---|
| *Lagocephalus sceleratus* | 378 | **182** | Altın test seti pozitifleri (yeni kaynak, eğitimde hiç görülmemiş) |
| *Lagocephalus suezensis* | 47 | 37 | Tür kararına göre pozitif veya zor negatif |
| *Lagocephalus guentheri* | 17 | 10 | 〃 |
| *Torquigener flavimaculosus* | 867 | 576 | 〃 (en sık karıştırılabilecek puffer) |
| *Siganus luridus* / *rivulatus* | 4428 | 3099 | Zor negatif (benzer boyut, aynı habitat) |
| *Sparisoma cretense* | 4907 | 3583 | Zor negatif |

Not: Yalnızca su altı ve su üstünden görünümlü fotoğraflar alınır (CLIP filtresi + insan doğrulaması). Mevcut veri setiyle pHash kopya kontrolü yapılır. Lisans her görüntü için kaydedilir.

## 6. Altın test seti tasarımı (öneri)

| Bileşen | Kaynak | Hedef adet |
|---|---|---|
| Pozitif, su altı ve su üstünden | Mevcut test + valid'in doğrulanmış "suda" görüntüleri (sızıntı hariç) + iNaturalist *L. sceleratus* | ≥ 150 |
| Zor negatif (başka türler, suda) | iNaturalist *Siganus*, *Sparisoma* (+ tür kararına göre diğer puffer'lar) | ≥ 80 |
| Boş sahne (balıksız su) | Mevcut negatifler + açık su altı veri setleri | ≥ 40 |
| Hareket bulanıklığı kopyası | Pozitiflerin sentetik bulanıklaştırılmış kopyası (ADR-000 K7) | Pozitif sayısı kadar |

- Kaynak düzeyinde ayrılır: aynı gözlem veya fotoğrafçı serisinden gelen görüntüler aynı sette kalır.
- Oluşturulduktan sonra **dondurulur**. Dosya listesi ve hash'leri `dataset_meta/gold_v1.csv` dosyasına yazılır.
- Su dışı fotoğraflar eğitimde kalabilir (Faz 2'de E1-b deneyiyle etkisi ölçülecek), ancak test setinde yer almaz.

## 7. Sıradaki adımlar
1. Ortam etiketlerinin doğrulanması (önce test + valid: 280 görüntü). Arayüz: `/denetim`.
2. Tür kapsamı kararı.
3. iNaturalist indirmesi (kullanıcı onayıyla), CLIP filtresi ve kopya kontrolü.
4. Yeni pozitifler için kutu etiketleme: modelle ön etiket + arayüzde düzeltme.
5. `gold_v1` oluşturulup dondurulması. Mevcut modelin bu sette ölçülmesi (E1-a, gerçek başlangıç noktası).
