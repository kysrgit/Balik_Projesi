# Devir Planı — Kalan İşler (çok ayrıntılı)

> Bu belge, projeye başka bir araçla (ör. Google Antigravity + Gemini) devam edebilmen için yazıldı.
> Her adımda **ne**, **neden**, **hangi komut**, **hangi dosya**, **başarı ölçütü** ve **tuzaklar** var.
> Tarih: 2026-10-20. Son commit: `git log -1` ile bak.

---

## 0. Şu an neredeyiz? (tek paragraf)

Faz 0 ve Faz 1 kapandı. Faz 2 neredeyse bitti.
- **Faz 0 — Gereksinimler:** [docs/adr/ADR-000-gereksinimler.md](adr/ADR-000-gereksinimler.md).
- **Faz 1 — Veri ve altın test seti:** [docs/adr/ADR-001-veri.md](adr/ADR-001-veri.md).
- **Faz 2 — Eğitim:** [docs/adr/ADR-002-egitim.md](adr/ADR-002-egitim.md).

En iyi tarif **E2-3**: mevcut train verisi + iNaturalist zor negatifleri + hareket bulanıklığı kopyaları; sentetik negatifler çıkarılmış. YOLO11s tek tohumla eşik 0,5'te şunları verdi:

| Ölçüt | Sonuç |
|---|---|
| Recall | %82,1 |
| Precision | %92,8 |
| Altın set negatif alarmı | %4,5 |
| Dış negatif alarmı | %2,7 |
| Bulanıklıkta recall kaybı | 3,8 puan |

Yani ölçülebilen bütün hedefler karşılanıyor. Şu anda arka planda **E3** çalışıyor: aynı tarifle YOLO11n, YOLO26n ve YOLO11s, 3'er tohum. Sonra sırasıyla şu fazlar gelecek:
- **Faz 3:** Pi'ye dışa aktarım (NCNN).
- **Faz 5:** İzleyici ve zamansal doğrulama.
- **Faz 6:** Yazılım temizliği.
- **Faz 7:** Sistem testleri.
- **Faz 8:** Saha veya dış video testi.
- **Faz 9:** Makale ve yayın.

Yol haritası: [docs/ROADMAP.md](ROADMAP.md). Derin araştırma raporu: [reports/Pi 5 su altı balık tespiti.md](../reports/Pi%205%20su%20alt%C4%B1%20bal%C4%B1k%20tespiti.md). Bu belgedeki önerilerin çoğu o rapordaki "Öncelikli eylem listesi"nden geliyor.

---

## 1. Ortam ve erişim bilgileri

### 1.1 PC (Windows 11, proje kökü `C:\AI\Balik_Projesi_Antigravity`)
| Ortam | Yol | Ne için | Not |
|---|---|---|---|
| `.venv` | `.venv\Scripts\python.exe` | Python 3.14, CPU. Testler, inceleme arayüzü, ONNX işleri | `pip` var |
| `.venv_gpu` | `.venv_gpu\Scripts\python.exe` | Python 3.11, **torch 2.6 + CUDA**. Eğitim, CLIP/OWLv2, GPU tahmin | **pip yok**, paket kurmak için: `uv pip install --python .venv_gpu\Scripts\python.exe <paket>` |
| GPU | RTX 3060 Laptop, **6 GB** | YOLO11s/n/26n 640 px, batch 8 sığıyor | 960 px + batch 8 sığmıyor |

Testler (her değişiklikten sonra):
```bash
.venv/Scripts/python.exe -m pytest tests/ -q -p no:cacheprovider
```
Beklenen: **103 passed**.

### 1.2 Raspberry Pi 5
- Ağ adı: `keysar.local`, kullanıcı: `keysar`, OS: Raspberry Pi OS Desktop 64-bit (Debian 13 trixie), Python 3.13.
- Parolasız SSH anahtarı: `~/.ssh/balik_pi` (yalnızca bu proje için). Örnek:
  ```bash
  ssh -i ~/.ssh/balik_pi keysar@keysar.local
  ```
- Proje Pi'de: `~/Balik_Projesi`, sanal ortam: `~/Balik_Projesi/.venv_pi`. Bu ortamda `ncnn`, `onnxruntime 1.31` ve `ultralytics 8.4` kurulu.
- **sudo parola istiyor.** Servis yeniden başlatma ve apt işlerini **sen** yapmalısın:
  ```bash
  ssh -t -i ~/.ssh/balik_pi keysar@keysar.local "sudo systemctl restart balik-dashboard"
  ```
- Dashboard servisi: `balik-dashboard` (systemd), port **5000** → http://keysar.local:5000
- Pi'ye kod gönderme yöntemi. Commit'ler GitHub'da olmayabilir, bu yüzden doğrudan git push ile gönderilir:
  ```bash
  GIT_SSH_COMMAND="ssh -i ~/.ssh/balik_pi" git push keysar@keysar.local:Balik_Projesi main:incoming
  ssh -i ~/.ssh/balik_pi keysar@keysar.local "cd ~/Balik_Projesi && git merge --ff-only incoming && git branch -D incoming"
  ```
- **Model dosyaları git'te YOK** (`*.pt`, `*.onnx`, `*_ncnn_model/` .gitignore'da). Ayrıca kopyalanmalı:
  ```bash
  scp -i ~/.ssh/balik_pi -r models/<dosya_veya_klasor> keysar@keysar.local:Balik_Projesi/models/
  ```

### 1.3 İnceleme arayüzü (PC)
```bash
.venv/Scripts/python.exe -m app.review.server
```
Adres: http://localhost:5050. Sayfalar:
- `/`: tespit inceleme; kutuya tıklayıp doğru/yanlış işaretlenebiliyor.
- `/denetim`: ortam doğrulama.
- `/etiket`: iNaturalist kutu etiketleme.

Port meşgulse `REVIEW_PORT=5052` ortam değişkeniyle başka port kullanılabilir.

### 1.4 GitHub
- Uzak depo: `https://github.com/kysrgit/Balik_Projesi.git`, dal `main`.
- **Push edilmemiş 18+ commit var.** Push etmeden önce kontrol:
  ```bash
  git log --oneline origin/main..HEAD
  ```
- ⚠️ Proje klasörü Google Drive ile senkron. Drive `.git/refs` içine `desktop.ini` koyuyor ve `git fetch` "bad object refs/desktop.ini" hatası veriyor. Çözüm (yalnızca `.git/refs` ve `.git/logs/refs` içindeki `desktop.ini`'ler silinir):
  ```bash
  find .git/refs .git/logs/refs -name desktop.ini -delete
  ```
  Kalıcı çözüm: proje klasörünü Drive senkronundan çıkarmak.

---

## 2. ŞU AN ÇALIŞAN İŞ: E3 eğitimleri

### 2.1 Ne çalışıyor
Arka planda 8 eğitim **sırayla** çalışıyor. Hepsi E2-3 verisiyle (`dataset_v2/E2-3_full.yaml`), aynı tarifle (`training/train_e2.py` içindeki `RECIPE`):

| Sıra | Model | Tohum | Çıktı klasörü |
|---|---|---|---|
| 1 | yolo11n.pt | 0 | `runs/e2/E2-3_full_yolo11n_s0` |
| 2 | yolo11n.pt | 1 | `runs/e2/E2-3_full_yolo11n_s1` |
| 3 | yolo11n.pt | 2 | `runs/e2/E2-3_full_yolo11n_s2` |
| 4 | yolo26n.pt | 0 | `runs/e2/E2-3_full_yolo26n_s0` |
| 5 | yolo26n.pt | 1 | `runs/e2/E2-3_full_yolo26n_s1` |
| 6 | yolo26n.pt | 2 | `runs/e2/E2-3_full_yolo26n_s2` |
| 7 | yolo11s.pt | 1 | `runs/e2/E2-3_full_s1` |
| 8 | yolo11s.pt | 2 | `runs/e2/E2-3_full_s2` |

YOLO11s tohum 0 zaten var: `runs/e2/E2-3_full_s0`. Toplam süre ~5-6 saat; nano ~30-35 dk, s ~45-50 dk.

Her eğitim bittiğinde betik otomatik olarak `scripts/eval_gold.py` çalıştırır ve şu dosyayı yazar:
`experiments/E2_gold_<ad>.json`. İçinde `gold`, `gold_blur` ve `gold_ext` kümeleri, 0,25/0,4/0,5/0,6 eşikleri ve bootstrap güven aralıkları var.

### 2.2 Takip etme
```bash
grep -E "^(=====|DONE|HATA)" experiments/E3_train.log
```
Anlık durum (son satır):
```bash
tail -c 400 experiments/E3_train.log | tr '\r' '\n' | tail -2
```
GPU durumu:
```bash
nvidia-smi
```

### 2.3 Durdurmak gerekirse
Görev Yöneticisi'nden `python.exe` (komut satırında `train_e2.py` olan) sonlandırılır. Yeniden başlatınca `--skip-existing` bayrağı **biten** eğitimleri atlar; yarım kalan eğitim baştan başlar:
```bash
.venv_gpu/Scripts/python.exe training/train_e2.py --variants E2-3_full --model yolo26n.pt --seed 1 --skip-existing
```
⚠️ Dizüstünün kapağı kapanırsa uyur ve eğitim durur.

### 2.4 E3 bitince yapılacak: sonuçları topla (ADIM A)
Aşağıdaki betiği `scripts/summarize_e3.py` olarak kaydet ve çalıştır. Her model için 3 tohumun ortalamasını ve standart sapmasını, her eşikte verir:

```python
"""E3: model x tohum sonuclarini ozetler (ortalama ± std)."""
import json, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
runs = {
    "YOLO11n": [f"E2-3_full_yolo11n_s{s}" for s in (0, 1, 2)],
    "YOLO26n": [f"E2-3_full_yolo26n_s{s}" for s in (0, 1, 2)],
    "YOLO11s": [f"E2-3_full_s{s}" for s in (0, 1, 2)],
}
ext_old = json.loads((EXP / "E2_gold_ext.json").read_text(encoding="utf-8"))["models"]  # s0 icin gold_ext burada

def load(name):
    d = json.loads((EXP / f"E2_gold_{name}.json").read_text(encoding="utf-8"))
    m = list(d["models"].values())[0]
    if "gold_ext" not in m:                       # E2-3_full_s0 eski formatta
        m["gold_ext"] = ext_old[f"{name}/best.pt"]["gold_ext"]
    return m

for t in ("0.4", "0.5", "0.6"):
    print(f"\n=== esik {t} ===")
    for model, names in runs.items():
        ms = [load(n) for n in names if (EXP / f"E2_gold_{n}.json").exists()]
        if not ms:
            continue
        def col(fn):
            v = [fn(m) for m in ms]
            return f"{st.mean(v)*100:5.1f}±{(st.stdev(v) if len(v) > 1 else 0)*100:4.1f}"
        print(f"{model:8s} n={len(ms)} "
              f"R={col(lambda m: m['gold'][f'conf_{t}']['recall']['value'])} "
              f"P={col(lambda m: m['gold'][f'conf_{t}']['precision']['value'])} "
              f"goldAlarm={col(lambda m: m['gold'][f'conf_{t}']['neg_alarm_rate']['value'])} "
              f"disAlarm={col(lambda m: m['gold_ext'][f'conf_{t}']['neg_alarm_rate']['value'])} "
              f"bulanikR={col(lambda m: m['gold_blur'][f'conf_{t}']['recall']['value'])} "
              f"kucukR={col(lambda m: m['gold'][f'conf_{t}']['small_recall']['value'])}")
```
Çalıştır:
```bash
.venv/Scripts/python.exe scripts/summarize_e3.py
```

### 2.5 Karar kuralı (ADIM B) — hangi model seçilecek
1. **Zorunlu:** Pi'de uçtan uca ≥ 10 FPS (ADR-000 K5). Bu yüzden **YOLO11s otomatik olarak elenir** (Pi'de 162 ms ≈ 6 FPS). YOLO11s yalnızca "doğruluk tavanı" referansıdır.
2. YOLO11n ile YOLO26n karşılaştırılır. Eşik olarak **0,5** (veya K1'i ≥ %80 tutan en yüksek eşik) kullanılır.
3. İki model arasındaki fark **tohum standart sapmasının 2 katından küçükse** "fark yok" kabul edilir. Bu durumda:
   - Pi'de daha hızlı olan seçilir. YOLO26n NCNN ~67 ms (Ultralytics), YOLO11n NCNN 70 ms (proje ölçümü). Pi'de aynı protokolle yeniden ölç (bkz. Faz 3).
   - Hız da eşitse **YOLO26n** seçilir. Gerekçe: NMS gerektirmiyor, bu yüzden çift kutu sorunu yok ve gecikme kuyruğu daha kısa.
4. Fark anlamlıysa (≥ 2 std), **dış alarm (K4) ve recall (K1)** öncelikli; sonra bulanık recall (K7).
5. Araştırma uyarısı: YOLO26 küçük veriyle daha zor öğrenebiliyor. YOLO26n'in recall'ü YOLO11n'den belirgin düşükse YOLO11n seçilir.
6. Kararı **ADR-002**'ye "Karar" bölümü olarak yaz. Tabloyu `summarize_e3.py` çıktısından doldur. Durumu "Kabul edildi" yap. `docs/ROADMAP.md` içindeki Faz 2 satırını ✅ yap.

---

## 3. Faz 3 — Seçilen modeli Pi'ye taşıma (NCNN)

**Neden NCNN:** Pi 5 CPU'sunda ölçülen en hızlı motor. FP16 aritmetiği donanımda kullanıyor (ONNX Runtime FP16'yı FP32'ye çeviriyor; proje ölçümü: kazanç yok). Kanıt: ADR-002 öncesi ölçümler ve araştırma raporu Faz 3.

### 3.1 En iyi tohumu seç
3 tohumdan, val_v2'de (eğitim sırasında) en iyi `fitness`'ı vereni seç. **Altın sete göre seçme!** Bu test setine bakarak seçim olur. Kolay yol: `runs/e2/<ad>/results.csv` içindeki en büyük `metrics/mAP50-95(B)` değeri.

### 3.2 NCNN'e dışa aktar
Dikdörtgen giriş önerisi (640×480): kamera 4:3 olduğu için letterbox boşluğu kalmaz, ~%25 daha hızlı olması beklenir.
```bash
.venv_gpu/Scripts/python.exe -c "from ultralytics import YOLO; YOLO('runs/e2/<SECILEN>/weights/best.pt').export(format='ncnn', imgsz=(480,640), half=False)"
```
Çıktı `runs/e2/<SECILEN>/weights/best_ncnn_model/` klasörüne yazılır. Bunu `models/pufferfish_v3_ncnn_model/` olarak kopyala.

⚠️ Dikdörtgen girişin altın sette doğruluğu bozmadığını kontrol et (3.3). Bozuyorsa `imgsz=640` (kare) kullan.

### 3.3 Doğruluk kontrolü (PC'de)
```bash
.venv/Scripts/python.exe scripts/eval_gold.py --sets gold gold_blur gold_ext --models models/pufferfish_v3_ncnn_model --out experiments/E3_ncnn_check.json
```
**Kabul:** Recall, alarm ve bulanık recall, `.pt` sonucundan en fazla 1 puan farklı olmalı.

### 3.4 Pi'de hız ölçümü
1. Modeli kopyala:
   ```bash
   scp -i ~/.ssh/balik_pi -r models/pufferfish_v3_ncnn_model keysar@keysar.local:Balik_Projesi/models/
   ```
2. Dashboard'u duraklat (ölçümü bozmasın). Önce PID'yi bul:
   ```bash
   ssh -i ~/.ssh/balik_pi keysar@keysar.local 'systemctl show -p MainPID --value balik-dashboard'
   ```
   Sonra durdur:
   ```bash
   ssh -i ~/.ssh/balik_pi keysar@keysar.local 'kill -STOP <PID>'
   ```
   Ölçüm bitince devam ettir:
   ```bash
   ssh -i ~/.ssh/balik_pi keysar@keysar.local 'kill -CONT <PID>'
   ```
3. Uçtan uca ölçüm. Pi'de `/tmp/ulbench.py` vardı; yoksa yeniden yaz: 40 test görüntüsü, 8 ısınma, 3 tur, `YOLO(model).predict(im, imgsz=..., conf=0.5)` medyanı.
   ```bash
   ssh -i ~/.ssh/balik_pi keysar@keysar.local 'cd ~/Balik_Projesi && . .venv_pi/bin/activate && python /tmp/ulbench.py models/pufferfish_v3_ncnn_model'
   ```
4. **Kabul:** medyan ≤ 100 ms, ve p95 ≤ 100 ms (ADR-000 K5).
5. Sonuçları `benchmark_pi5/` altına JSON olarak kaydet ve commit'le.

### 3.5 (İsteğe bağlı) NCNN INT8
Araştırmaya göre NCNN INT8, FP16'ya göre yalnızca %13-19 hızlı. Bunu ancak 3.4 hedefi tutmazsa dene. Kural: **tespit başı FP32 kalmalı**; kalibrasyon sualtı karelerinden yapılmalı. Doğruluğu **bulanık kümede de** kontrol et.

### 3.6 Uygulamayı yeni modele bağla
`app/core/config.py` dosyasında:
```python
MODEL_PATH = ROOT_DIR / "models" / "pufferfish_v3_ncnn_model"
CONF_THRESH = 0.50          # ADR-002: K1'i tutan eşik; E3 sonucuna göre güncelle
DETECTOR_IMGSZ = 640        # dikdörtgen kullanıyorsan detector.py'de imgsz=(480, 640) ver
```
- `app/core/detector.py` → `self.model.predict(... imgsz=config.DETECTOR_IMGSZ ...)`. `ultralytics.YOLO('<ncnn klasoru>')` NCNN'i doğrudan açar; başka değişiklik gerekmez.
- `detector.py` başındaki `PatchedInferenceSession` (ONNX Runtime yaması) NCNN için gereksiz ama zararsız. Temizlik istersen kaldır ve testleri çalıştır.
- **NMS IoU'yu 0,5'e indir** (YOLO11n seçildiyse; YOLO26n'de NMS yok): `predict(..., iou=0.5, agnostic_nms=True)`. Çift kutu sorunu (fotograf_504) bununla azalır.
- `tests/` içinde model yoluna bağlı testler varsa güncelle; 103 test geçmeli.
- Commit → Pi'ye git push (1.2) → modeli scp ile kopyala → servisi yeniden başlat (sudo, sen).

---

## 4. Faz 5 — Zamansal doğrulama ve küçük nesne

**Önemli sıra kuralı (araştırma raporu):** "5 karenin 3'ünde görülme" kuralı, kare başına başarım iyiyse hatayı azaltır, kötüyse **artırır**. E2-3 ile kare başına alarm ~%3-5 olduğu için artık uygulanabilir.

### 4.1 ByteTrack + 3/5 kuralı
1. `app/dashboard/server.py` → `detection_loop()` içinde `detector.detect()` yerine ultralytics izleyicisini kullan:
   ```python
   results = self.model.track(source=img, persist=True, tracker="bytetrack.yaml", conf=0.1, iou=0.5, verbose=False)
   ```
   Düşük `conf`, ikinci aşama eşleştirme için gerekli; asıl eşik iz düzeyinde uygulanır.
2. Her `track_id` için son 5 karedeki görülme sayısını tut (`collections.deque(maxlen=5)`). **≥ 3** ise ve ortalama güven ≥ `CONF_THRESH` ise alarm üret. Aynı iz için tek kayıt yaz (CSV, veritabanı, webhook).
3. ReID ve kamera hareketi telafisini kapat: kendi `bytetrack.yaml` dosyanı kopyala ve `gmc_method: none` yap. Kare başına maliyet birkaç ms.
4. **Test:** Saha videosu yok. Altın set görüntülerinden sentetik diziler üret: her görüntüden kaydırma, ölçek ve bulanıklıkla 10 kare. Kare ve iz düzeyinde recall ile alarmı ayrı raporla. Yeni betik: `scripts/eval_temporal.py`.

### 4.2 Bulanıklık skoru
Laplasyen varyansı (`cv2.Laplacian(gray, cv2.CV_64F).var()`), maliyeti ~1 ms. Çok bulanık karede yeni iz başlatma; mevcut izleri sürdür.

### 4.3 Küçük nesne
- **Önce ölçülebilir yap:** gold_v1'de yalnızca 13 küçük balık var. Hedef: **≥ 100 küçük nesne** içeren ek bir test kümesi.
  - Kaynak: iNaturalist'te etiketlenmemiş *Torquigener* fotoğrafları (5-11 cm, uzaktan çekilenler).
  - Etiketleme: `scripts/propose_boxes.py` (YOLO + OWLv2) önerir, `/etiket` sayfasında onaylanır.
  - Bu kümeyi `gold_v1_small` olarak ayrı dondur; gold_v1'i **değiştirme**.
- Tam kare SAHI **yapma**: Pi'de ~2 FPS'e düşürür. Bunun yerine izleyicinin "düşük güvenli iz" bildirdiği bölgeden saniyede 1 kez yüksek çözünürlüklü kırpma (ROI) al.

### 4.4 CLAHE
Çıkarımda **kapalı kalsın** (`DETECT_USE_CLAHE = False`). İstersen eğitimde rastgele CLAHE artırması dene. Başarı ölçütü: gold F1 düşmemeli.

---

## 5. Faz 6 — Yazılım iyileştirmeleri (öncelik sırasıyla)

1. **eventlet → threading:** eventlet bakımda değil. `app/dashboard/server.py` içinde `SocketIO(app, async_mode="threading")` yap. `eventlet.tpool` ve `socketio.sleep` kullanımlarını `time.sleep` ve threading ile değiştir. `EVENTLET_ENABLED` ayarını kaldır. Testleri güncelle (`tests/test_dashboard_stats.py` tpool'u mock'luyor).
2. **Picamera2 iki akış:** `main` (1280×720) kayıt için, `lores` (640×480 **RGB**, Pi 5'te mümkün) çıkarım için. Dosya: `app/core/camera.py`.
3. **Olay kaydı:** `CircularOutput2` ile tespitten önceki ve sonraki birkaç saniyeyi kaydet. Bu kayıtlar ileride zor negatif madenciliği ve saha test seti için altın değerinde.
4. **Telemetri satırı (her kare):** kare no, UTC zamanı, GPS, kutular ve skorlar, iz kimliği, SoC sıcaklığı, `vcgencmd get_throttled`. JSON Lines olarak `data/telemetry/` altına yaz. Faz 7 ölçümleri buradan çıkacak.
5. **Isı koruması:** SoC > 75 °C ise tespit sıklığını düşür (kare atla).
6. **systemd watchdog:** `scripts/balik-dashboard.service` dosyasına `WatchdogSec=30` ekle. Uygulama `sdnotify` ile ping atsın. Pi 5'in önyükleyici watchdog'u işletim sistemi açıldıktan sonra çalışmıyor.
7. **Kurulum betiği:** `scripts/install_pi.sh` dosyasına `ncnn` ekle (`requirements.txt` içine `ncnn` yaz).

Her madde ayrı commit; her commit'te 103+ test geçmeli.

---

## 6. Faz 7 — Sistem testleri

| Test | Nasıl | Kabul |
|---|---|---|
| Model regresyonu | `scripts/eval_gold.py` ile seçilen modeli ölç; sonuçları `experiments/regression_baseline.json` olarak sakla; yeni modelde K1/K3/K4/K7 düşmemeli | Hedefler korunur |
| Pi uçtan uca gecikme | Dashboard çalışırken (kamera takılıyken) telemetriden p50/p95/p99 | p95 ≤ 100 ms |
| **1 saat termal** | Kapalı muhafaza (veya muhafaza benzeri kutu) içinde, havada (en kötü durum), tam boru hattı. Her 10 sn'de `vcgencmd measure_temp` ve `get_throttled` | `throttled=0x0`, sıcaklık < 80 °C |
| 8 saat dayanıklılık | Bellek (RSS) artışı, disk dolması (`MAX_*_FILES` sınırları) | Bellek sabit, disk sınırlı |
| Video tekrar testi | Kamera yerine kayıtlı video ver (OpenCV `VideoCapture(dosya)`); beklenen tespitlerle karşılaştır | Kare/iz metrikleri raporlanır |

Kamera takılı değilken dashboard CPU'su %10 civarı olmalı (önceden %260'tı, düzeltildi). Bu bir regresyon testine çevrilebilir.

---

## 7. Donanım önerileri (araştırma raporundan)

- **Kamera:** Raspberry Pi **Global Shutter Camera (IMX296)**, M12/CS lens, 2,8-4 mm. Hareketli platformda yuvarlanan deklanşör eğikliği olmaz. Ucuz alternatif: Camera Module 3 **Wide**, manuel odakla.
- **Pozlama:** `AeEnable=False`, `ExposureTime=1000-2000` µs, kazanç serbest, beyaz dengesi sabit, HDR/TDN kapalı. 1,3 m/s göreli hızda 2 ms pozlama ≈ 2 piksel bulanıklık.
- **Muhafaza:** Alüminyum kapak veya tüp. SoC → termal ped → alüminyum plaka → suya giden ısı yolu. Kapalı kutuda fan yalnızca havayı karıştırır.
- **Güç:** Pi ~8,8 W, sistem 10-13 W. 5 V/5 A; buck dönüştürücüyle `PSU_MAX_CURRENT=5000`. Teknede güç + Ethernet kablosu (tether) en basit çözüm.
- **Düz port** görüş açısını su içinde ~1,33 kat daraltır; geniş lens seç.

---

## 8. Faz 8 — Saha ya da dış video doğrulaması

1. Saha videosu yoksa: **lisansı uygun kamuya açık sualtı videolarından** bir dış video test kümesi kur. Eğitimde hiç kullanılmamış olmalı. Kaynaklar: Wikimedia Commons, CC lisanslı YouTube videoları (lisansı kontrol et), LILA veri seti videoları.
2. Her video için: kaç puffer geçti (insan sayımı), sistem kaç iz buldu, saat başına yanlış alarm.
3. **Raporlama kare değil olay (iz) düzeyinde** olmalı. Test verisi video veya saha bazında bölünmeli, aynı videodan rastgele kare alınmamalı.
4. Saha erişimi olursa en iyi pencere **Haziran, Güneybatı Türkiye koyları, 5-10 m** (*L. sceleratus* üreme kümelenmesi).
5. İnsan doğrulamalı iş akışı: model önerir, uzman onaylar. İnceleme arayüzü (`/`) bunun için uyarlanabilir.

---

## 9. Faz 9 — Yayın ve teslim

1. **Makale:** `archive/manuscript/manuscript_EN.md` dosyası eski sayılara (YOLO11m/s, eski test seti) dayanıyor. **Tüm sayılar ADR-001 ve ADR-002'deki gold_v1 sonuçlarıyla değiştirilmeli.** Eski test setinin yarısının su dışında olduğu ve sentetik negatiflerin bulunduğu dürüstçe yazılmalı.
2. **Katkılar:** (1) taradığımız literatürde *L. sceleratus* için ilk görüntü tabanlı tespit; (2) Pi 5 CPU'da ≥ 10 FPS; (3) zor negatif ve bulanıklık ablasyonu, dış negatif kontrolüyle birlikte.
3. **Model kartı:** `docs/MODEL_CARD.md`. İçerik: eğitim verisi, lisanslar, beklenen başarım, sınırlılıklar (saha videosu yok, küçük nesne n=13, etiketlerin bir kısmı Claude tarafından verildi).
4. **Veri lisansı:** Türetilmiş veri seti yayımlanacaksa yalnızca CC0, CC BY, CDLA-permissive ve MIT kaynakları kullanılmalı. iNaturalist fotoğraflarının lisansı `dataset_meta/inat_photos.csv` içinde; CC BY-NC olanlar ticari değil.
5. **README güncelle:** yeni model, yeni sayılar, NCNN kurulum adımı.
6. **GitHub'a push** (1.4'teki desktop.ini notuna dikkat).

---

## 10. ASLA yapılmaması gerekenler (tuzaklar)

1. **`dataset_gold_v1`, `dataset_gold_v1_blur`, `dataset_gold_v1_ext` ve bunların `dataset_meta/*.csv` dosyalarını değiştirme.** Değişiklik gerekirse `gold_v2` oluştur.
2. **Model veya eşik seçimini altın sete bakarak yapma.** Model seçimi val_v2 ile (eğitim sırasında) yapılır. Altın set yalnızca son ölçümdür. Eşik seçimi de mümkünse val_v2'de yapılmalı; ADR-000 bunu öngörüyor.
3. Altın setteki görüntülerin kopyalarını (pHash grubu) veya aynı iNaturalist gözlemini eğitime koyma. `scripts/build_trainset.py` bunu zaten engelliyor; yeni veri eklerken de bu kuralı koru.
4. Çıkarımda CLAHE açma. ONNX Runtime FP16 kullanma (hız kazancı yok).
5. Eğitim ve çıkarım çözünürlüğünü farklı tutma.
6. Pi'de `sudo` gerektiren işleri parolasız yapmaya çalışma; kullanıcıya bırak.
7. Kullanıcının açtığı inceleme sunucusunu habersiz kapatma (kayıtları yarıda kesilir).
8. Sentetik negatifleri (`neg_synth_*`) geri ekleme: dış testte zarar verdiği görüldü.

---

## 11. Dosya haritası (önemli olanlar)

| Yol | Ne |
|---|---|
| `docs/ROADMAP.md` | 10 fazlı yol haritası ve durum tablosu |
| `docs/adr/ADR-000…002` | Kararlar (gereksinim, veri, eğitim) |
| `docs/DATA_AUDIT.md` | Veri denetimi raporu |
| `reports/Pi 5 su altı balık tespiti.md` | Derin araştırma raporu (kaynaklı) |
| `dataset/` | Orijinal Roboflow verisi (git dışı) |
| `dataset_inat/` | iNaturalist indirmeleri (git dışı); meta: `dataset_meta/inat_photos.csv` |
| `dataset_ext/imedea/` | IMEDEA Mallorca indirmesi (git dışı) |
| `dataset_gold_v1*/` | Dondurulmuş test setleri (git dışı); listeler `dataset_meta/gold_v1*.csv` (git'te) |
| `dataset_v2/` | Faz 2 eğitim varyantları (liste dosyaları, git dışı) |
| `dataset_meta/` | Denetim, etiket, düzeltme kayıtları (git'te) |
| `scripts/data_audit.py` | pHash kopya, ortam (CLIP), etiket istatistikleri |
| `scripts/fetch_inat.py` | iNaturalist indirme |
| `scripts/propose_boxes.py` | YOLO + OWLv2 bağımsız kutu önerisi |
| `scripts/build_gold.py`, `build_gold_ext.py` | Altın setleri oluştur ve dondur |
| `scripts/build_trainset.py` | Eğitim varyantları + val_v2 |
| `scripts/eval_gold.py` | ADR-000 ölçütleriyle değerlendirme (`--sets gold gold_blur gold_ext`) |
| `training/train_e2.py` | Sabit tarifli eğitim + otomatik değerlendirme |
| `scripts/benchmark_pi5.py` | Pi 5 gecikme ölçümü |
| `app/review/` | İnceleme, denetim ve etiketleme arayüzü |
| `app/core/config.py` | Model yolu, eşik, CLAHE ayarı |
| `experiments/` | Tüm deney sonuçları (JSON) ve loglar |
| `runs/e2/` | Eğitim çıktıları ve ağırlıklar (git dışı) |

---

## 12. Gemini / Antigravity için başlangıç mesajı (kopyala-yapıştır)

> Bu proje TÜBİTAK 2209-B kapsamında istilacı balon balıklarının (L. sceleratus, L. suezensis, L. guentheri, Torquigener flavimaculosus — tek sınıf) Raspberry Pi 5 CPU üzerinde su altında tespiti. Proje kökü `C:\AI\Balik_Projesi_Antigravity`. Lütfen önce şu dosyaları sırayla oku: `docs/DEVIR_PLANI.md` (bu devir planı), `docs/ROADMAP.md`, `docs/adr/ADR-000-gereksinimler.md`, `docs/adr/ADR-001-veri.md`, `docs/adr/ADR-002-egitim.md`, ve `reports/Pi 5 su altı balık tespiti.md`. Kurallar: altın test setlerini (`dataset_gold_v1*`) asla değiştirme; model/eşik seçimini altın sete bakarak yapma; her kararı ölçümle ver ve bir ADR'ye yaz; her değişiklikten sonra `.venv/Scripts/python.exe -m pytest tests/ -q -p no:cacheprovider` ile 103 testin geçtiğini doğrula; Pi'de sudo gerektiren işleri bana bırak. İlk görev: `docs/DEVIR_PLANI.md` bölüm 2.4–2.5'e göre E3 eğitim sonuçlarını özetle ve model kararını ADR-002'ye yaz.
