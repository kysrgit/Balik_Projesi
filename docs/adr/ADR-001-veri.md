# ADR-001: Tespit kapsamı ve veri kaynakları

**Durum:** Kısmen kabul edildi (kapsam ve kaynaklar kabul; veri sürümü `gold_v1` oluşturulunca tamamlanacak)
**Tarih:** 2026-10-15
**Faz:** 1
**Karar verenler:** Kayra (proje sahibi)

## Bağlam
Veri denetimi ([DATA_AUDIT.md](../DATA_AUDIT.md)) iki sorun gösterdi:
1. Test setinin yaklaşık %40'ı su dışı fotoğraf; negatif yok, 1 sızıntı var.
2. Etiket sınıfı genel olarak `pufferfish`. Veri setinde *L. sceleratus* dışındaki puffer türleri de bulunuyor.

ADR-000'a göre saha verisi yok; yalnızca açık kaynak kullanılacak.

## Karar
1. **Kapsam: tüm istilacı (Lesepsiyen) puffer türleri tek sınıf** (`pufferfish`, "balon balığı"). Pozitif kabul edilen türler: *Lagocephalus sceleratus*, *L. suezensis*, *L. guentheri*, *Torquigener flavimaculosus*. Mevcut etiketlerle uyumlu.
2. **Zor negatifler:** Aynı habitatta yaşayan, benzer boyutta puffer olmayan türler. Başlangıç seti: *Siganus luridus*, *Siganus rivulatus*, *Sparisoma cretense*.
3. **Açık kaynak:** iNaturalist, yalnızca araştırma kalitesinde ve açık lisanslı fotoğraflar (CC0, CC BY, CC BY-SA, CC BY-NC, CC BY-NC-SA). Her görüntünün lisansı, fotoğrafçısı ve gözlem bağlantısı `dataset_meta/inat_photos.csv` dosyasına kaydedilir. Görüntüler `dataset_inat/` klasörüne iner (git dışında).
4. **Ortam:** Altın test seti yalnızca "suda" görüntülerden oluşur: su altı ve su üstünden görünüm. Su dışı fotoğraflar test setine alınmaz.

## Değerlendirilen seçenekler
| Seçenek | Artı | Eksi |
|---|---|---|
| Yalnızca *L. sceleratus* | Tür düzeyinde kesin kayıt | Mevcut etiketlerin yeniden incelenmesi gerekir; daha az veri |
| **Tüm istilacı puffer'lar (seçildi)** | Mevcut etiketlerle uyumlu, daha çok veri, izleme amacına uygun | Tür ayrımı yapılmaz |
| İki sınıf | En bilgilendirici | En çok etiketleme emeği |

## Sonuçları
- Model "puffer" ile "puffer olmayan balık" ayrımını öğrenmek zorunda. Zor negatifler bunu ölçer ve öğretir.
- Tür düzeyinde ayrım gerekirse ileride ikinci aşama bir sınıflandırıcı eklenebilir.
- CC BY-NC lisanslı görüntüler araştırma ve yayın için kullanılabilir; veri seti ticari olarak dağıtılamaz.

## Yapılacaklar
1. [ ] iNaturalist indirmesi (`scripts/fetch_inat.py`)
2. [ ] Ortam etiketlerinin doğrulanması (`/denetim`)
3. [ ] Yeni pozitifler için kutu etiketleme
4. [ ] `gold_v1` oluşturulup dondurulması
