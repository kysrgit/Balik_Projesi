/**
 * Balon Balığı Tespit Sistemi — Doğrulanmış Proje Veri Kataloğu
 *
 * Bu dosya, proje dokümantasyonu, ADR belgeleri ve deney çıktılarından doğrulanan
 * tür, veri seti, benchmark, kabul ölçütü, API uç noktası ve galeri verilerini
 * `window.BALIK_PROJECT_DATA` altında merkezi olarak dışa aktarır.
 */
(function (globalScope) {
  'use strict';

  /**
   * Tüm web arayüzü bileşenlerinin paylaştığı doğrulanmış proje veri nesnesi.
   * Değerler yalnızca depo içi ölçüm ve ADR kayıtlarından alınmıştır.
   */
  const BALIK_PROJECT_DATA = {
    project: {
      name: 'Balon Balığı Tespit Sistemi',
      program: 'TÜBİTAK 2209-B',
      repository: 'https://github.com/kysrgit/Balik_Projesi.git',
      summary:
        'Doğu Akdeniz ve Türkiye kıyılarında istilacı Lesepsiyen balon balığı türlerinin Raspberry Pi 5 üzerinde gerçek zamanlı sualtı tespiti ve coğrafi etiketli (GIS) izlenmesi.',
      targetUsers: [
        'Deniz biyologları',
        'Su ürünleri araştırmacıları',
        'İstilacı tür izleme ve yönetim ekipleri',
      ],
      phases: [
        { id: 'faz-0', label: 'Faz 0 — Gereksinimler (ADR-000)', status: 'completed' },
        { id: 'faz-1', label: 'Faz 1 — Veri ve Gold v1 (ADR-001)', status: 'completed' },
        { id: 'faz-2', label: 'Faz 2 — Veri Tarifi ve E3 Deneyleri (ADR-002)', status: 'in-progress' },
        { id: 'faz-3', label: 'Faz 3 — Pi 5 NCNN Doğrulama', status: 'partial' },
        { id: 'faz-4', label: 'Faz 4 — AI HAT+ Hızlandırıcı', status: 'skipped' },
        { id: 'faz-5-9', label: 'Faz 5–9 — Saha, Akış ve Paketleme', status: 'planned' },
      ],
      testSuite: {
        testFileCount: 14,
        passingRoadmap: 103,
        passingReadme: 102,
      },
    },

    species: {
      classCount: 1,
      classNames: ['pufferfish'],
      items: [
        {
          id: 'lagocephalus-sceleratus',
          latinName: 'Lagocephalus sceleratus',
          commonName: 'Gümüş Yanlı Balon Balığı',
          category: 'invasive',
          role: 'Hedef Pozitif Tür (pufferfish)',
          notes:
            'Doğu Akdeniz kıyılarında yüksek yayılım gösteren başlıca istilacı Lesepsiyen balon balığı türü.',
        },
        {
          id: 'lagocephalus-suezensis',
          latinName: 'Lagocephalus suezensis',
          commonName: 'Süveyş Balon Balığı',
          category: 'invasive',
          role: 'Hedef Pozitif Tür (pufferfish)',
          notes:
            'Sığ kıyı sularında ve ROV/tekne gözlem derinliklerinde görülen istilacı Lesepsiyen türü.',
        },
        {
          id: 'lagocephalus-guentheri',
          latinName: 'Lagocephalus guentheri',
          commonName: 'Günther Balon Balığı',
          category: 'invasive',
          role: 'Hedef Pozitif Tür (pufferfish)',
          notes:
            'Türkiye ve Doğu Akdeniz sularında tespit edilen istilacı Tetraodontidae üyesi.',
        },
        {
          id: 'torquigener-flavimaculosus',
          latinName: 'Torquigener flavimaculosus',
          commonName: 'Sarı Benekli Cüce Balon Balığı',
          category: 'invasive',
          role: 'Hedef Pozitif Tür (pufferfish)',
          notes:
            '5–11 cm boyu nedeniyle küçük nesne tespitinde kritik zorluk oluşturan tür; Haziran ayında 5–10 m derinlikte üreme kümelenmesi gösterir.',
        },
        {
          id: 'siganus-luridus',
          latinName: 'Siganus luridus',
          commonName: 'Esmer Sokar',
          category: 'hard-negative',
          role: 'Zor Negatif Tür (iNaturalist)',
          notes:
            'Aynı kıyı habitatını paylaşan hedef dışı tür; E1-a başlangıcında yanlış alarmların ana kaynaklarından biridir.',
        },
        {
          id: 'siganus-rivulatus',
          latinName: 'Siganus rivulatus',
          commonName: 'Beyaz Sokar',
          category: 'hard-negative',
          role: 'Zor Negatif Tür (iNaturalist)',
          notes:
            'Eğitim setinde bulunmadığında yüksek yanlış pozitif üreten Akdeniz resif türü.',
        },
        {
          id: 'sparisoma-cretense',
          latinName: 'Sparisoma cretense',
          commonName: 'Iskaroz (Papağan Balığı)',
          category: 'hard-negative',
          role: 'Zor Negatif Tür (iNaturalist)',
          notes:
            'Renk ve vücut formu nedeniyle zor negatif setine dahil edilen Akdeniz kıyı balığı.',
        },
        {
          id: 'imedea-mallorca-ext',
          latinName: 'DEEPECOMAR / IMEDEA Mallorca',
          commonName: 'Dış Negatif Akdeniz Balık Seti (25 Sınıf)',
          category: 'dataset',
          role: 'Dış Negatif Doğrulama (Zenodo 7534425)',
          notes:
            'CC BY 4.0 lisanslı, eğitimde hiç görülmemiş 25 Akdeniz türünü içeren 300 pHash-ayıklanmış negatif görüntü (297 görüntüsünde farklı balık türleri bulunur).',
        },
      ],
    },

    datasets: {
      summary: {
        rawPhotosTotal: 827,
        roboflowTrain: 1879,
        roboflowValid: 197,
        roboflowTest: 83,
        inatDownloaded: 1744,
        inatPhashDuplicates: 452,
        goldV1Total: 327,
        goldV1Positive: 215,
        goldV1Negative: 112,
        goldV1Instances: 235,
        goldV1SmallInstances: 13,
        goldV1BlurTotal: 327,
        goldV1ExtTotal: 300,
      },
      rawByRegion: [
        { region: 'Greece', count: 300 },
        { region: 'Diger', count: 245 },
        { region: 'Turkey', count: 136 },
        { region: 'Cyprus', count: 74 },
        { region: 'Israel', count: 38 },
        { region: 'Spain', count: 19 },
        { region: 'Egypt', count: 5 },
        { region: 'Syria', count: 5 },
        { region: 'Italy', count: 3 },
        { region: 'Lebanon', count: 2 },
      ],
      roboflowV1: {
        id: 'balon-baligi-tespiti-oa6kg',
        license: 'CC BY 4.0',
        resolution: '640x640',
        splits: {
          train: { images: 1879, uniqueSources: 725, positive: 1728, negative: 151, instances: 2217 },
          valid: { images: 197, positive: 164, negative: 33, syntheticNegative: 13, lowQualityBg: 6, instances: 183 },
          test: {
            images: 83,
            positive: 82,
            negative: 1,
            instances: 139,
            environmentClip: { land: 32, underwater: 22, surface: 29 },
            knownLeak: 'test:fotograf_71 ≈ train:fotograf_81',
          },
        },
      },
      goldV1Composition: {
        total: 327,
        positiveImages: 215,
        negativeImages: 112,
        fishInstances: 235,
        smallObjectsUnder1Pct: 13,
        sources: [
          { label: 'dataset/test (suda)', count: 39 },
          { label: 'dataset/valid (suda)', count: 96, detail: '84 pozitif + 12 boş sahne' },
          { label: 'iNaturalist pozitif', count: 92, detail: '70 su altı + 22 su üstü' },
          { label: 'iNaturalist zor negatif', count: 100, detail: 'Siganus / Sparisoma' },
        ],
      },
      v2Variants: [
        { id: 'val_v2', file: 'dataset_v2/val_v2.txt', images: 145, note: 'Kaynak düzeyinde ayrılmış 93 pozitif + 52 zor negatif' },
        { id: 'E2-0_base', file: 'dataset_v2/E2-0_base.txt', images: 1611, note: 'Temel eğitim seti (val_v2 ve gold pHash grupları hariç)' },
        { id: 'E2-1_hardneg', file: 'dataset_v2/E2-1_hardneg.txt', images: 1907, note: 'base + 296 iNaturalist zor negatif' },
        { id: 'E2-2_blur', file: 'dataset_v2/E2-2_blur.txt', images: 2420, note: 'base + 809 hareket-bulanık kopya (k=5–21)' },
        { id: 'E2-3_full', file: 'dataset_v2/E2-3_full.txt', images: 2645, note: 'base - 52 sentetik negatif + zor negatifler + bulanık kopyalar' },
      ],
    },

    adr000Criteria: [
      { id: 'K1', title: 'Genel Recall', target: '>= %80', measuredE23At05: '%82,1 (YOLO26n E3: %85,5)', met: true },
      { id: 'K2', title: 'Küçük Nesne Recall (alan < %1)', target: '>= %50', measuredE23At05: 'YOLO11s E2-3: %46,2 (n=13) / YOLO26n E3: %51,3', met: true },
      { id: 'K3', title: 'Precision', target: '>= %70', measuredE23At05: '%92,8 (YOLO26n E3: %94,1)', met: true },
      { id: 'K4', title: 'Boş Sahne / Negatif Alarm Oranı', target: '<= %5', measuredE23At05: 'Gold %4,5 / Dış %2,7', met: true },
      { id: 'K5', title: 'Pi 5 Gerçek Zamanlı Hız', target: '>= 10 FPS (p95 <= 100 ms)', measuredE23At05: 'ONNX INT8 5,85 FPS (171,07 ms) / NCNN FP16 70 ms (~14 FPS)', met: true },
      { id: 'K6', title: '1 Saat Termal Kararlılık', target: 'Throttling ve bellek sızıntısı yok', measuredE23At05: '200 karelik kısa koşuda throttled=0x0 (63,90–75,45 °C); 1 saatlik kararlılık testi Faz 3 kapsamında planlı', met: false, status: 'pending' },
      { id: 'K7', title: 'Bulanık Kopyada Recall Kaybı', target: '<= 10 puan', measuredE23At05: '3,8 puan kayıp (%82,1 -> %78,3)', met: true },
    ],

    benchmarks: {
      pcCpu: {
        hardware: 'Intel Core i7-11800H (Windows 11, Python 3.14.3, ONNX Runtime 1.24.4 CPU)',
        trainingGpu: 'RTX 3060 Laptop 6 GB (PyTorch GPU referans gecikmesi: 8,80 ms)',
        datasetSplit: 'dataset/test (83 görüntü, 15 ısınma, conf=0,60)',
        models: [
          { name: 'PyTorch (.pt)', sizeMb: 19.18, meanMs: 118.48, medianMs: 117.62, p90Ms: 123.12, fps: 8.44 },
          { name: 'ONNX FP32', sizeMb: 37.93, meanMs: 98.30, medianMs: 98.02, p90Ms: 101.15, fps: 10.17 },
          { name: 'ONNX INT8 Karma', sizeMb: 12.53, meanMs: 56.89, medianMs: 55.28, p90Ms: 60.91, fps: 17.58, validMeanMs: 54.01, validFps: 18.51 },
        ],
      },
      pi5Cpu: {
        hardware: 'Raspberry Pi 5 Model B Rev 1.1 (16.218 MB RAM, Active Cooler, Debian 13 trixie, ORT 1.31.0 CPU)',
        protocol: '640x640, CLAHE kapalı, 20 ısınma + 200 kare (ön işleme + çıkarım + NMS)',
        onnxModels: [
          { name: 'pufferfish_s_v2_int8.onnx', sizeMb: 12.53, meanMs: 171.07, p95Ms: 171.66, fps: 5.85, tempRangeC: '53,45 -> 63,90', throttled: '0x0', testMap50: 0.7452, validAlarm: '0/33' },
          { name: 'pufferfish_s_v2.onnx (FP32)', sizeMb: 37.93, meanMs: 362.13, p95Ms: 364.00, fps: 2.76, tempRangeC: '59,50 -> 71,05', throttled: '0x0' },
          { name: 'cand_s_int8.onnx', sizeMb: 12.53, meanMs: 173.28, p95Ms: 175.38, fps: 5.77, tempRangeC: '54,55 -> 65,55', throttled: '0x0' },
          { name: 'cand_s_fp16.onnx', sizeMb: 19.02, meanMs: 368.06, p95Ms: 370.48, fps: 2.72, tempRangeC: '57,30 -> 68,30', throttled: '0x0' },
          { name: 'cand_s.onnx (FP32)', sizeMb: 37.93, meanMs: 365.40, p95Ms: 366.30, fps: 2.74, tempRangeC: '57,85 -> 72,70', throttled: '0x0' },
          { name: 'cand_m_int8.onnx', sizeMb: 25.13, meanMs: 374.76, p95Ms: 375.66, fps: 2.67, tempRangeC: '57,85 -> 70,50', throttled: '0x0' },
          { name: 'cand_m_fp16.onnx', sizeMb: 40.28, meanMs: 1373.36, p95Ms: 1393.58, fps: 0.73, tempRangeC: '58,40 -> 72,70', throttled: '0x0' },
          { name: 'cand_m.onnx (FP32)', sizeMb: 80.43, meanMs: 998.49, p95Ms: 1001.23, fps: 1.00, tempRangeC: '58,40 -> 75,45', throttled: '0x0' },
        ],
        ncnnReported: [
          { name: 'YOLO11n NCNN (FP16, 640 px)', latencyMs: 70, approxFps: 14, testMap50: 0.7382, validAlarm: '0/33', meetsK5: true },
          { name: 'YOLO11s NCNN (FP16, 640 px)', latencyMs: 162, approxFps: 6, testMap50: 0.7494, validAlarm: '0/33', meetsK5: false },
        ],
      },
      experiments: {
        e1aBaseline: {
          model: 'pufferfish_s_v2_int8.onnx',
          conf: 0.60,
          goldRecallPct: 87.66,
          goldPrecisionPct: 73.31,
          hardNegAlarmPct: 49.11,
          smallRecallPct: 46.15,
          goldMap50: 0.8624,
          blurRecallPct: 45.96,
          blurMap50: 0.6004,
        },
        e2Recipe: [
          { id: 'E2-0 base', conf: 0.4, trainMin: 28.0, recallPct: 90.6, precisionPct: 64.0, goldAlarmPct: 58.9, extAlarmPct: 36.7, blurRecallPct: 49.8, blurDropPts: 40.8, map50: 0.8579 },
          { id: 'E2-1 hardneg', conf: 0.4, trainMin: 32.6, recallPct: 83.4, precisionPct: 92.9, goldAlarmPct: 5.4, extAlarmPct: 15.0, blurRecallPct: 51.9, blurDropPts: 31.5, map50: 0.9150 },
          { id: 'E2-2 blur', conf: 0.4, trainMin: 43.1, recallPct: 90.2, precisionPct: 63.1, goldAlarmPct: 56.2, extAlarmPct: 42.7, blurRecallPct: 83.0, blurDropPts: 7.2, map50: 0.8479 },
          { id: 'E2-3 full', conf: 0.4, trainMin: 46.8, recallPct: 85.5, precisionPct: 89.7, goldAlarmPct: 5.4, extAlarmPct: null, blurRecallPct: 80.4, blurDropPts: 5.1, map50: 0.9088 },
          { id: 'E2-3 full (@0,5)', conf: 0.5, trainMin: 46.8, recallPct: 82.1, precisionPct: 92.8, goldAlarmPct: 4.5, extAlarmPct: 2.7, blurRecallPct: 78.3, blurDropPts: 3.8, map50: 0.9088, smallRecallPct: 46.2 },
        ],
        e3MultiSeed: [
          { arch: 'YOLO11n (n=3: s0, s1, s2)', recallPct: '81,4 ± 2,6', precisionPct: '92,9 ± 2,0', goldAlarmPct: '3,3 ± 1,4', extAlarmPct: '5,4 ± 2,8', blurRecallPct: '77,0 ± 1,1', smallRecallPct: '43,6 ± 4,4', map50Pct: '90,4 ± 0,8' },
          { arch: 'YOLO26n (n=3: s0, s1, s2)', recallPct: '85,5 ± 1,1', precisionPct: '94,1 ± 1,2', goldAlarmPct: '3,6 ± 0,9', extAlarmPct: '6,8 ± 0,8', blurRecallPct: '82,0 ± 1,2', smallRecallPct: '51,3 ± 8,9', map50Pct: '92,2 ± 0,4' },
          { arch: 'YOLO11s (n=2: s0, s1)', recallPct: '83,6 ± 2,1', precisionPct: '94,5 ± 2,4', goldAlarmPct: '3,6 ± 1,3', extAlarmPct: '3,5 ± 1,2', blurRecallPct: '78,7 ± 0,6', smallRecallPct: '50,0 ± 5,4', map50Pct: '91,1 ± 0,4' },
          { arch: 'YOLO11m (n=1: s0)', recallPct: '83,0', precisionPct: '95,1', goldAlarmPct: '3,6', extAlarmPct: '5,3', blurRecallPct: '76,6', smallRecallPct: '38,5', map50Pct: '89,2' },
        ],
      },
    },

    application: {
      runtimeConfig: {
        modelPath: 'models/pufferfish_s_v2_int8.onnx',
        confThresh: 0.60,
        detectorImgsz: 640,
        camResolution: '640x480',
        targetFps: 30,
        skipFrames: 5,
        detectUseClahe: false,
        claheClip: 3.0,
        gpsPort: '/dev/ttyAMA0',
        gpsBaudrate: 9600,
        gpioLedPin: 17,
        dashboardPort: 5000,
        reviewPort: 5050,
        streamFps: 15,
      },
      dashboardEndpoints: [
        { method: 'GET', path: '/', description: 'Canlı operasyon ve debug dashboard arayüzü' },
        { method: 'GET', path: '/video/<stream_type>', description: 'raw, clahe veya detection MJPEG video akışı' },
        { method: 'GET/POST', path: '/api/config', description: 'Güven eşiği, CLAHE clip ve overlay ayarları' },
        { method: 'POST', path: '/api/snapshot', description: 'Anlık kare kaydı (DASHBOARD_API_TOKEN korumalı)' },
        { method: 'POST', path: '/api/record', description: 'Video kayıt başlat / durdur kontrolü' },
        { method: 'GET', path: '/api/export/geojson', description: 'QGIS ve Leaflet uyumlu GeoJSON dışa aktarımı' },
        { method: 'GET', path: '/api/export/csv', description: 'Araştırmacılar için tespit günlüğü CSV çıktısı' },
        { method: 'GET', path: '/api/export/darwincore', description: 'GBIF / OBIS uyumlu DarwinCore Archive (ZIP)' },
        { method: 'GET/POST/DEL', path: '/api/webhooks', description: 'SSRF korumalı Slack / Discord / Teams bildirim yönetimi' },
      ],
      reviewEndpoints: [
        { method: 'GET', path: '/', description: 'GT vs tahmin kutuları ve canlı P/R/F1 inceleme ekranı' },
        { method: 'POST', path: '/api/run', description: 'Model ve CLAHE seçimiyle anlık çıkarım karşılaştırması' },
        { method: 'POST', path: '/api/upload', description: 'Görsel veya video yükleme (en fazla 120 kare)' },
        { method: 'POST', path: '/api/fix', description: 'Uzman onaylı kutu düzeltme kaydı' },
        { method: 'GET', path: '/denetim', description: 'Ortam doğrulama (/api/audit, /api/audit/tag)' },
        { method: 'GET', path: '/etiket', description: 'iNaturalist kutu etiketleme (/api/label/list, /api/label/save)' },
      ],
    },

    gallery: [
      {
        id: 'kapak',
        category: 'detection',
        src: './assets/kapak.jpg',
        title: 'Sualtı Balon Balığı Kapak Karesi',
        caption: 'dataset_gold_v1 içerisinden alınan gerçek sualtı balon balığı gözlem karesi (1920x1080).',
        meta: ['Kaynak: dataset_gold_v1', 'Çözünürlük: 1920x1080', 'Sınıf: 0 (pufferfish)'],
      },
      {
        id: 'cozum-tespit',
        category: 'detection',
        src: './assets/cozum_tespit.jpg',
        title: 'Gerçek Zamanlı Tespit Çıktısı',
        caption: '640x640 letterbox ön işleme sonrası sınır kutusu ve güven skoru ile balon balığı tespiti.',
        meta: ['Giriş: 640x640 letterbox', 'Eşik: conf 0,60', 'Model: YOLO11s INT8'],
      },
      {
        id: 'tespit-ornekleri',
        category: 'detection',
        src: './assets/tespit_ornekleri.jpg',
        title: 'Çoklu Sualtı Tespit Örnekleri',
        caption: 'Farklı ışık, derinlik ve açı koşullarında doğrulanan tespit örnekleri kolajı.',
        meta: ['Altın Set: 327 görüntü', 'E2-3 mAP@50: 0,9088', 'YOLO26n E3: %92,2 ± 0,4'],
      },
      {
        id: 'int8-karsilastirma',
        category: 'detection',
        src: './assets/int8_karsilastirma.jpg',
        title: 'FP32 vs Karma Hassasiyet INT8 Karşılaştırması',
        caption: 'Tespit başı FP32’de bırakılarak backbone/neck QDQ INT8 nicemlemesiyle korunan tespit doğruluğu.',
        meta: ['Pi 5 INT8: 171,07 ms (5,85 FPS)', 'Pi 5 FP32: 362,13 ms (2,76 FPS)', 'Boyut: 12,53 MB'],
      },
      {
        id: 'veri-kirliligi',
        category: 'data',
        src: './assets/veri_kirliligi.jpg',
        title: 'Ortam Dağılımı ve Veri Kirliliği Analizi',
        caption: 'Roboflow v1 test setindeki 83 görüntünün CLIP ortam dağılımı: 32 kara, 29 su üstü, 22 su altı.',
        meta: ['Test seti: 83 görüntü', 'Su altı oranı: 22/83', 'Kara kareleri: 32/83'],
      },
      {
        id: 'phash-kopya',
        category: 'data',
        src: './assets/phash_kopya.jpg',
        title: 'pHash Kopya ve Sızıntı Tespiti',
        caption: 'iNaturalist ve Roboflow splitleri arasında pHash ile tespit edilen 452 çakışan görüntü ve sızıntı örneği.',
        meta: ['iNat çakışması: 452/1.744', 'Sızıntı örneği: test:fotograf_71 ≈ train:fotograf_81', 'Araç: imagehash'],
      },
      {
        id: 'zor-negatif-turler',
        category: 'data',
        src: './assets/zor_negatif_turler.jpg',
        title: 'Aynı Habitattaki Zor Negatif Türler',
        caption: 'Siganus luridus, Siganus rivulatus ve Sparisoma cretense türlerinden oluşan 100 zor negatif örneği.',
        meta: ['Gold zor negatif: 100 görüntü', 'Dış set: 300 IMEDEA görüntüsü', '25 Akdeniz türü'],
      },
      {
        id: 'zor-negatif-once-sonra',
        category: 'data',
        src: './assets/zor_negatif_once_sonra.jpg',
        title: 'Zor Negatif Eğitimi Öncesi ve Sonrası',
        caption: 'E2-0 temel veri ile E2-3 tam tarifin gold_v1_ext üzerindeki yanlış alarm düşüşü (%32,0 -> %2,7 @0,5).',
        meta: ['E1-a alarm: %49,11', 'E2-3 gold alarm: %4,5', 'E2-3 dış alarm: %2,7'],
      },
      {
        id: 'bulaniklik-once-sonra',
        category: 'data',
        src: './assets/bulaniklik_once_sonra.jpg',
        title: 'Hareket Bulanıklığı Dayanıklılığı (E2-0 vs E2-3)',
        caption: 'Sabit tohumlu gold_v1_blur testinde recall kaybının 40,8 puandan 3,8 puana indirilmesi.',
        meta: ['E1-a bulanık R: %45,96', 'E2-3 bulanık R: %78,3 (@0,5)', 'K7 hedefi: <= 10 puan kayıp'],
      },
      {
        id: 'canli-panel',
        category: 'ui',
        src: './assets/canli_panel.png',
        title: 'Canlı Web Dashboard (Port 5000)',
        caption: 'Flask-SocketIO, MJPEG akışı, Leaflet heatmap haritası, donanım telemetrisi ve DarwinCore/GeoJSON aktarımı.',
        meta: ['Port: 5000', 'Akış: raw / clahe / detection / WS', 'Harita: Leaflet + heatmap'],
      },
      {
        id: 'arayuz-inceleme',
        category: 'ui',
        src: './assets/arayuz_inceleme.png',
        title: 'Model İnceleme Arayüzü (/)',
        caption: 'GT ve tahmin kutularını yan yana karşılaştıran, canlı P/R/F1 ve boş sahne alarmı hesaplayan inceleme aracı.',
        meta: ['Sunucu: python -m app.review.server', 'Port: 5050', 'Özellik: Eşik ve CLAHE testi'],
      },
      {
        id: 'arayuz-denetim',
        category: 'ui',
        src: './assets/arayuz_denetim.png',
        title: 'Ortam Doğrulama Ekranı (/denetim)',
        caption: 'Görüntülerin su altı, su üstü ve kara ortam etiketlerinin uzman tarafından doğrulandığı denetim arayüzü.',
        meta: ['Rota: /denetim', 'API: /api/audit, /api/audit/tag', 'Ön etiket: CLIP'],
      },
      {
        id: 'arayuz-etiket',
        category: 'ui',
        src: './assets/arayuz_etiket.png',
        title: 'Kutu Etiketleme Ekranı (/etiket)',
        caption: 'iNaturalist kaynaklı sualtı görüntülerinde uzman onaylı sınır kutusu etiketleme arayüzü.',
        meta: ['Rota: /etiket', 'API: /api/label/list, /api/label/save', 'İlke: Model önerir, uzman onaylar'],
      },
    ],
  };

  if (typeof globalScope !== 'undefined') {
    globalScope.BALIK_PROJECT_DATA = BALIK_PROJECT_DATA;
  }
})(typeof window !== 'undefined' ? window : globalThis);
