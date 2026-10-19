# Detector Architectures and Training Strategies for Single-Class Underwater Pufferfish Detection on Raspberry Pi 5 (state as of Oct 2026)

Context for the report writer: target is a single class (invasive Lagocephalus / Torquigener pufferfish), ~725 unique training photos (~2k files after augmentation), deployment on Raspberry Pi 5 CPU at >=10 FPS, moving ROV/boat platform, recall is the priority. The team has observed YOLO11m generalising worse than YOLO11s on their data. Research budget was limited (~17 tool calls), so several sub-questions below are only partially covered. These are flagged under Gaps.

## 1. Architecture comparison: YOLO11 / YOLO26 / YOLOv8-v12 / RT-DETR family / other lightweight detectors on underwater benchmarks and edge CPUs

### Takeaway
YOLO26n is currently the strongest nano-scale option for a Pi 5 CPU. Ultralytics reports YOLO26n at 40.9 COCO mAP50-95, 2.4M params and 5.5 GFLOPs, ~15 FPS on a Pi 5 via NCNN at 640 px (model inference only). On underwater benchmarks, recent YOLO generations (v8 to v12, YOLO11, YOLO26) land within a few mAP points of each other. RT-DETR is usually equal or slightly better on accuracy and recall but is not Pi-5-viable at 10 FPS. Gains from published "underwater-specific YOLO variants" are self-reported, measured against inconsistent baselines, and come with conflicting baseline numbers, so treat them sceptically.

### Cited Findings
- **YOLO26 (released January 2026), key changes:** an optional NMS-free one-to-one head (`nms=False`; the default path still uses NMS); Distribution Focal Loss (DFL) removed to simplify the head and export; MuSGD optimizer (SGD + Muon hybrid); Progressive Loss (ProgLoss); STAL (Small-Target-Aware Label Assignment), which "improves positive label coverage for small objects". — [Ultralytics YOLO26 docs](https://docs.ultralytics.com/models/yolo26/)
- **YOLO26 COCO numbers at 640 px.** The "e2e" column is the NMS-free mode. CPU times are ONNX on Ultralytics' test CPU. — [Ultralytics YOLO26 docs](https://docs.ultralytics.com/models/yolo26/)

  | Model | mAP50-95 | mAP50-95 (e2e) | CPU ONNX ms | Params (M) | FLOPs (B) |
  |---|---|---|---|---|---|
  | YOLO26n | 40.9 | 40.1 | 38.9 | 2.4 | 5.5 |
  | YOLO26s | 48.6 | 47.8 | 87.2 | 9.5 | 20.9 |
  | YOLO26m | 53.1 | 52.5 | 220.0 | 20.4 | 68.4 |

- **Speed claim:** Ultralytics claims "up to 43% faster CPU ONNX inference" for YOLO26n vs YOLO11n, measured on an Intel Xeon @ 2.00 GHz. The docs give no quantified small-object claim for STAL. — [Ultralytics YOLO26 docs](https://docs.ultralytics.com/models/yolo26/)
- **Raspberry Pi 5, 640 px input, FP32 (vendor numbers):**
  - YOLO26n: NCNN 67.69 ms/image (~14.8 FPS), ONNX 130.33 ms (~7.7 FPS).
  - YOLO26n vs YOLO11n with ONNX: 6.79 to 7.79 FPS (~15% gain), with mAP 40.1 vs 39.5.
  - Older table (Ultralytics 8.3.39): YOLO11n NCNN 88.0 ms, ONNX 168.1 ms.
  - Times exclude pre/post-processing. The YOLO26 table headers name two Ultralytics versions (8.4.14 and 8.4.1), so the runs may not be identical.

  — [Ultralytics Raspberry Pi guide](https://docs.ultralytics.com/guides/raspberry-pi)
- **YOLO26 on RUOD (10-class general underwater benchmark).** Reported numbers conflict between papers:
  - The MDP-YOLO paper (Scientific Reports 2026) reports YOLO26 at 83.7% mAP@0.5 and 49.5% mAP@0.5:0.95, vs 86.7% mAP@0.5 for MDP-YOLO.
  - A related table reports YOLO26 at 84.6% mAP@0.5 and 61.3% mAP@0.5:0.95 with 4.6M params.

  — [MDP-YOLO, Sci Rep / ResearchGate](https://www.researchgate.net/publication/413958768_MDP-YOLO_underwater_object_detection_based_on_enhanced_feature_extraction); [Nature reference PDF](https://www.nature.com/articles/s41598-026-69925-1_reference.pdf). These come from a search-result summary; I did not read the tables directly. The 12-point mAP50-95 discrepancy suggests different training setups or splits.
- **RT-DETR vs YOLO on RUOD (MDP-YOLO comparison):** RT-DETR reached 81.6% recall on RUOD, higher than MDP-YOLO's own 79.8%. YOLOv8n, YOLOv10n and YOLO11n fall in the same mAP range in that comparison. — [MDP-YOLO (ResearchGate)](https://www.researchgate.net/publication/413958768_MDP-YOLO_underwater_object_detection_based_on_enhanced_feature_extraction)
- **Domain-aware benchmark (Wille, Miller, Fischer, Raine; QUT; arXiv 2607.10575, 2026):**
  - Setup: YOLO26n, Faster R-CNN and RT-DETR, each trained 50 epochs with default hyperparameters on DUO + RUOD-R + UTDAC2020 (18,852 images, 196,102 annotations).
  - Overall ranking: RT-DETR > YOLO > Faster R-CNN, consistently across domains.
  - Per-domain mAP50 gaps (best vs worst condition) are similar across architectures:
    - visibility ~0.28
    - colour ~0.33
    - scale (small vs large) ~0.21; YOLO26n 0.21, RT-DETR 0.26
    - illumination ~0.11

  So no architecture removes the low-visibility or small-object penalty. — [arXiv 2607.10575](https://arxiv.org/html/2607.10575)
- **YOLOv8 to YOLOv12 robustness (Nabahirwa et al., arXiv 2509.17561, Sep 2025):** 10,000 images from DUO + Roboflow100, six simulated underwater distortions. YOLOv12 gave the best overall performance but is "highly vulnerable to noise". Image counts and instance frequency drive detection performance more than object appearance does. — [arXiv 2509.17561](https://arxiv.org/abs/2509.17561)
- **YOLOv8-s to YOLOv11-s on underwater coral (4,480 images) and fish (7,500 images, 20 classes) datasets:** at 25/50/75/100% data fractions, "accuracy saturates after YOLOv9". Newer versions mainly improve speed. The lightweight YOLOv10 was recommended for embedded AUV use. — [Hung & Rodriguez, arXiv 2509.12682](https://arxiv.org/abs/2509.12682)
- **YOLOv12 sizes on a real-world underwater dataset:** YOLOv12-nano is 5.7% less accurate than YOLOv12-medium, with 2.57M params and 6.5 GFLOPs. On CPU, nano runs at 16.48 FPS vs 2.36 FPS for medium. — [Telematika journal, Comparative Study of Lightweight YOLOv12 Models](https://journal.ithb.ac.id/index.php/telematika/article/view/799) (via search summary)
- **Fish-specific (DeepFish + OzFish, 8,275 images):** YOLOv11m reached 78.3% mAP, while YOLOv12s reached 77.6% at about half the model size. — search summary of a 2026 study, [openaccess.ahievran.edu.tr](https://openaccess.ahievran.edu.tr/bitstreams/132a4ca7-40f5-430a-a311-b3696523552c/download) (I did not verify this attribution in full text)
- **Underwater-specific YOLO variants, DUO results (self-reported):**
  - AGS-YOLO: 85.5% mAP@0.5 and 67.7% mAP@0.5:0.95, framed as a low-resource small-object detector. — [AGS-YOLO, JMSE 2025](https://doi.org/10.3390/jmse13081465)
  - DyAqua-YOLO: 91.8% mAP@0.5 and 72.2% mAP@0.5:0.95. — search summary; also see [YOLO-DAFS, JMSE 2025](https://doi.org/10.3390/jmse13050947)

  Neither is benchmarked against YOLO26 under a common protocol.
- **Lightweight fish detector:** Foc_YOLOXn_ASFF reached 0.877 AP with 2.27M params. A plain YOLOX-Nano underfit: its "overly minimal model parameters fail to fit the training data, leading to significant missed detections". — [Aquacultural Engineering 2025, improved YOLOX-nano](https://www.sciencedirect.com/science/article/abs/pii/S0144860925000226)
- **Transformer vs CNN on sonar (UATD, not optical):** RT-DETR-L reached 0.535 mAP50-95 vs 0.507 for YOLOv8n and 0.481 for YOLOv12n. — [ResearchGate, RT-DETR vs YOLO on UATD](https://www.researchgate.net/publication/396820256_Transformer_vs_CNN_Benchmarking_RT-DETR_and_YOLO_Models_on_the_UATD_Dataset)
- **Ultralytics RUOD benchmark page:** lists YOLO26x at 49.85 and YOLO26l at 49.72 mAP50-95 on RUOD. The same table has a "YOLO27l" row, likely an error or placeholder. — [Ultralytics benchmark](https://docs.ultralytics.com/datasets/detect/ultralytics-benchmark)
- **Benchmark datasets:** DUO has 7,782 images in 4 classes; RUOD has 14,000 images in 10 classes, one of which is "fish". A survey recommends both as fair large-scale benchmarks. — [Underwater Object Detection in the Era of AI survey, arXiv 2410.05577 / ACM CSUR](https://arxiv.org/html/2410.05577v1)

### Inferences
- **Size choice.** For >=10 FPS end-to-end on a Pi 5 CPU, YOLO26n with NCNN is the only clearly viable Ultralytics model at 640 px: ~68 ms inference, plus pre/post-processing and camera I/O, likely puts end-to-end near 10-13 FPS. YOLO26s has ~3.8x the FLOPs (20.9 vs 5.5 B), so at 640 px it probably runs ~4-5 FPS. YOLO26s becomes plausible only at a reduced input size (e.g. 416 or below) or with INT8 quantisation. These are extrapolations, not measurements; benchmark on the actual Pi.
- **NMS-free head.** YOLO26's NMS-free head removes a post-processing step and its variable latency on CPU. This also matters for the duplicate-box issue noted in the project's roadmap. Its e2e mode costs ~0.7-0.8 COCO mAP points.
- **Architecture choice is secondary.** Since accuracy "saturates after YOLOv9" and domain gaps (visibility, colour, scale) are ~20-33 mAP points for every architecture, data and labels matter more than architecture choice for this project.
- **RT-DETR family.** RT-DETR is consistently best on accuracy and recall, so it is useful as a desktop teacher, reference model or auto-labeler. It is not suitable as the Pi-5 runtime model.

### Gaps
- No primary-source underwater benchmarks found for RT-DETRv2, D-FINE, DEIM, NanoDet, PicoDet, EfficientDet-Lite or MobileNet-SSD on RUOD, DUO, Brackish, DeepFish or FishNet, and no Pi 5 latency figures for these models.
- No independent (non-Ultralytics) Pi 5 benchmark of YOLO26n vs YOLO11n was found.
- No published detection work on pufferfish (Lagocephalus or Torquigener) was found. A search for lionfish/pufferfish YOLO detection returned only generic fish detection work.
- Conflicting YOLO26 RUOD numbers (49.5 vs 61.3 mAP50-95) could not be reconciled without reading both full papers.

## 2. Small-dataset strategies: COCO transfer vs underwater pretraining, freezing, image size, images needed, label quality

### Takeaway
With ~500-1,000 images, COCO pretraining is a safe default but did not beat training from scratch in the one controlled fish study found. Domain-matched fish pretraining (OzFish) gave a significant gain of +4 mAP points at 500 images. Badly matched marine datasets actively hurt. Ultralytics' own guidance (>=1,500 images and >=10,000 instances per class) means ~725 photos is well below the "comfortable" regime, which makes label completeness and consistency critical.

### Cited Findings
- **Ayyagari et al., ICES J. Mar. Sci. 2025 (YOLOv7):** fine-tuned on 500 and 1,000 frames of a custom underwater fish set, with 5 repeats each. Pretraining source effects:

  | Pretraining | mAP at 500 images | mAP at 1,000 images |
  |---|---|---|
  | None | 0.854 | 0.879 |
  | MS COCO | 0.857 (n.s.) | 0.879 (n.s.) |
  | OzFish (only 1,230 images, but closely domain-matched) | **0.896** (significantly higher) | 0.898 |
  | Norfisk | 0.794 (lower) | 0.858 |
  | NPSNF (18k images) | 0.655 (lower) | 0.824 (lower) |

  - **Convergence:** OzFish converged in 27±9 epochs vs 244 for COCO and 362 for scratch, at 500 images.
  - **Recommendation:** pick pretraining data that matches the target in resolution, imaging conditions and multi-fish framing. If you cannot test several sources, default to COCO or ImageNet. "More data is not automatically better."
  - **Out-of-distribution (Brackish):** every model collapsed (mAP ≤0.124).

  — [Ayyagari et al. 2025, ICES JMS 82(4) fsaf039](https://academic.oup.com/icesjms/article/82/4/fsaf039/8105847)
- **COCO has no fish class** (flagged 2018 work, older), which raises doubts about the value of COCO weights for fish detection. — [arXiv 1811.01494 (2018)](https://arxiv.org/pdf/1811.01494)
- **Freezing:** a 2026 fish scale-loss study (YOLOv8m) pretrained on above-water images, then fine-tuned only the neck and detection head on a few underwater samples. It reports that this helps under small-sample conditions. — [Fishes 11(1):21, 2026](https://doi.org/10.3390/fishes11010021) (via search summary)
- **Ultralytics dataset guidance:** "≥ 1500 images per class recommended"; "≥ 10000 instances (labeled objects) per class recommended"; "All instances of all classes in all images must be labeled"; "Labels must closely enclose each object". — [Ultralytics training tips](https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results/)
- **Ultralytics on pretraining and image size:** pretrained weights are "Recommended for small to medium-sized datasets". "If there are many small objects then custom datasets will benefit from training at native or higher resolution". Inference should use the same image size as training. — [Ultralytics training tips](https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results/)
- **Label quality (RUOD vs RUOD-R re-annotation):**
  - Object count rose from 34,433 to 72,982. Original labels mainly *missed* objects (scallop +294%, holothurian +71%).
  - Missed labels concentrate in small objects and crowded scenes (correction rate ~4x higher than for large objects and sparse scenes), dark images (2.9 vs 1.1), low visibility (2.3 vs 1.2) and green-tinted images (2.0 vs 0.9).
  - Authors recommend extra verification passes in hard domains, and preprocessing (contrast/colour) of hard images *before* labelling.

  — [Wille et al., arXiv 2607.10575](https://arxiv.org/html/2607.10575)
- **Data quantity:** image counts and instance frequency drive per-class performance more than appearance does. — [Nabahirwa et al., arXiv 2509.17561](https://arxiv.org/abs/2509.17561)

### Inferences
- **Concrete experiment:** compare (a) COCO-pretrained YOLO26n/s against (b) the same model first fine-tuned on a public fish set (OzFish, DeepFish, or Fish4Knowledge-style imagery that resembles ROV footage), then fine-tuned on the pufferfish data. Expect gains only if the source domain resembles the target. Mismatched sources can lower mAP by 6-20 points (Norfisk, NPSNF).
- **Missed boxes hurt recall.** Missed boxes on small, dark, turbid fish teach the model that those fish are background, which directly suppresses recall. A targeted second-pass audit of small/dark/turbid images in the ~725-photo set is likely to give more than any architecture change.
- **Effective dataset size.** The ~2k files after augmentation are not independent samples. Effective dataset size is ~725, roughly half Ultralytics' recommended minimum.

### Gaps
- No controlled study found on layer freezing (e.g. `freeze=10`) for YOLO11 or YOLO26 on small fish datasets.
- No study found that quantifies how many images per class are needed specifically for single-species fish detection.
- The ICES result is from YOLOv7 (2022 architecture). Whether it transfers to YOLO26 is unverified.

## 3. Hard negative mining: non-target fish species and background images to reduce false positives

### Takeaway
No fish-specific hard-negative study was found. Evidence from other single-target YOLO domains shows large FP reductions from adding background/negative images: 20% was optimal for river debris, and raising from 10% to 60% cut FPs by over 80% for roadwork. Ultralytics' default advice of 0-10% background images is conservative. For a species-specific detector, negatives should be visually confusable non-target fish taken from diverse scenes.

### Cited Findings
- **Ultralytics guidance:** "We recommend about 0-10% background images to help reduce FPs". COCO has ~1% background images. — [Ultralytics training tips](https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results/)
- **Roadwork detection (YOLO11m):** raising background images from 10% to 60% of training data reduced FPs by over 80% and *also* cut false negatives by 25%, measured on a test set with 60% background images. — [arXiv 2604.02282](https://arxiv.org/pdf/2604.02282)
- **Floating debris in urban rivers (YOLOv8n and YOLO11m):** several negative-image proportions were tested. 20% gave the best trade-off between detection performance and FP reduction for both architectures. — [arXiv 2510.23798](https://arxiv.org/pdf/2510.23798)
- **Practitioner report:** hard negatives mined from one video fixed FPs on that video, but FPs reappeared on new videos. Negatives must therefore span the scene diversity seen at deployment. — [Ultralytics community thread](https://community.ultralytics.com/t/yolo-v8-consider-the-background-as-a-target-what-can-i-do-about-that/1710)

### Inferences
- **Starting ratio.** For a single-class pufferfish detector, a reasonable starting point is 15-30% negatives (i.e. ~100-300 images for ~725 positives). Prioritise:
  - (a) other Mediterranean fish, especially other Tetraodontidae, and similar fusiform/silver fish with confusable shape or colour;
  - (b) empty water column, seabed, turbid and blurry frames from the actual ROV/boat camera.

  Then sweep the ratio (e.g. 10/20/40%) and measure recall *and* FPs on a gold test set that itself contains negatives. Because recall is the priority, watch for recall drops at high negative ratios. The roadwork study saw FN improve, but this is domain-dependent.
- **Labelled non-target alternative.** An alternative to pure background is a second "other fish" class: label non-target fish and keep only the pufferfish class at deploy time. No evidence for or against this was found in this search.

### Gaps
- No fish- or species-specific hard-negative ablation was found. The ratios above come from roadwork and debris detection and are extrapolated.
- No evidence found comparing "background-only negatives" vs "labelled non-target species as a second class".

## 4. Robustness to blur, turbidity, low light, colour cast: augmentation, synthetic data, domain gaps

### Takeaway
Underwater conditions (low visibility, colour cast) cost ~28-33 mAP50 points for every detector architecture, so robustness must come from data. Evidence is thin but consistent:
- Injecting degraded (noise-aware) samples in training improves robustness on real images.
- Enhancement-based fine-tuning helps only on enhanced inputs and slightly hurts on raw ones, so avoid making the pipeline depend on image enhancement.
- Copy-paste and diffusion synthesis give modest gains (~+3 mAP in the one quantified underwater case).

### Cited Findings
- **Domain gaps (mAP50, worst vs best condition), YOLO26n / RT-DETR / Faster R-CNN on 18.8k underwater images:**
  - colour: 0.33 / 0.31 / 0.33
  - visibility: 0.28 / 0.27 / 0.29
  - illumination: 0.10 / 0.10 / 0.12
  - Blue-tinted images scored highest (>30 points above "natural" colour). The authors attribute this to correlation with high visibility.

  — [Wille et al., arXiv 2607.10575](https://arxiv.org/html/2607.10575)
- **Nabahirwa et al. 2025:**
  - Noise disrupts edge and texture features and explains much of YOLO's poor performance on noisy images.
  - "Noise-aware sample injection" improves robustness on noisy *and* real-world images.
  - Fine-tuning with advanced enhancement raises accuracy on enhanced images but slightly lowers it on original images.

  — [arXiv 2509.17561](https://arxiv.org/abs/2509.17561)
- **Diffusion-generated underwater training images (Communications Engineering 2025):** YOLOv12 mAP improved by 3.2%, and mAP50 for "jellyfish" by 3.3%. The authors call the gain modest but practically meaningful. — [Nature Communications Engineering, s44172-025-00579-z](https://www.nature.com/articles/s44172-025-00579-z)
- **Copy-paste for fish:** an ACM 2025 paper evaluates copy-pasting fish instances to enlarge and diversify training data for YOLO variants. Numbers were not retrieved. — [ACM DL 10.1145/3759023.3759122](https://dl.acm.org/doi/10.1145/3759023.3759122)
- **Annealing-based copy-paste:** a 2026 preprint uses copy-paste-inspired, annealing-based augmentation on DeepFish to synthesise crowded fish scenes, which DeepFish underrepresents. — [arXiv 2604.21198](https://arxiv.org/abs/2604.21198)
- **Risk of diffusion + paste without masks:** pasted patches carry non-matching backgrounds (finding from an aerial-detection study). — [arXiv 2311.12345](https://arxiv.org/pdf/2311.12345)
- **Synthetic-only training:** a structured review describes a DALL-E 3 + OpenCV pipeline for synthetic underwater images, evaluated with YOLO11 on synthetic-only vs mixed data. — [arXiv 2509.08490](https://arxiv.org/pdf/2509.08490)
- **Out-of-distribution transfer:** all fish detectors (YOLOv7) trained on one site collapsed on Brackish (mAP ≤0.124). This shows how severe domain shift is between underwater sites and cameras. — [Ayyagari et al. 2025](https://academic.oup.com/icesjms/article/82/4/fsaf039/8105847)

### Inferences
- **Recommended augmentation recipe:** add degradation augmentations that mimic the deployment camera: motion blur, Gaussian noise, defocus, hue shift toward green/blue, contrast reduction or haze for turbidity, brightness reduction. Apply them stochastically to *raw* images; do not train or infer only on enhanced images.
- **Validating synthetic data.** Copy-paste of segmented pufferfish onto real underwater backgrounds, including real hard-negative backgrounds, is the cheapest synthetic-data step. Validate it with an ablation on the frozen gold set, since gains in the literature are modest (~3 points).
- **Out-of-water photos.** Above-water photos (e.g. iNaturalist catches on deck) are a domain gap. Brackish OOD collapse and the "domain-matched pretraining wins" result suggest weighting toward in-water imagery. Keep out-of-water images as a minority, or use them only in a first pretraining stage, with the final fine-tune on in-water images.

### Gaps
- No quantitative ablation found for Mosaic, MixUp or CutMix on small single-class underwater sets.
- No quantitative ablation found for motion-blur augmentation specifically.
- No study found that directly measures the effect of mixing out-of-water fish photos into underwater training.
- The six corruption types in Nabahirwa et al. and their per-corruption numbers were not retrieved (the PDF was too large to fetch).

## 5. Small-object improvements usable at nano scale (P2 head, higher resolution, SAHI-style crops) and latency costs

### Takeaway
YOLO26's STAL label assignment is a built-in, zero-inference-cost small-object aid. P2-head YAML configs exist for YOLO26 but have no released pretrained weights. Small objects trail large ones by >20 mAP50 points across detectors. No Pi-5 latency figures were found for P2 heads, larger input sizes or SAHI.

### Cited Findings
- **YOLO26 small-object support:** STAL improves positive label coverage for small objects. P2 (small-object) head variants exist only as YAML configs with no released weights. — [Ultralytics YOLO26 docs](https://docs.ultralytics.com/models/yolo26/)
- **Scale gap:** small objects trail large objects by over 20 mAP50 points: YOLO26n 0.21, RT-DETR 0.26, Faster R-CNN 0.17. — [Wille et al., arXiv 2607.10575](https://arxiv.org/html/2607.10575)
- **Resolution guidance:** custom datasets with many small objects "will benefit from training at native or higher resolution". — [Ultralytics training tips](https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results/)
- **Tiny fish remain hard:** a 2024 student study found transfer learning helps overall, but the best model still struggled on tiny fish. — [CS231n 2024 report](https://cs231n.stanford.edu/2024/papers/transfer-learning-for-fish-detection-in-underwater-images.pdf) (student project, low evidential weight)
- **Lightweight small-object detectors:** AGS-YOLO is positioned as a lightweight small-object underwater detector for low-resource environments (DUO 85.5 mAP@0.5). — [JMSE 2025](https://doi.org/10.3390/jmse13081465)

### Inferences
- **Compute cost.** CNN FLOPs scale roughly with pixel count, so going from 640 to 800 px costs ~1.56x compute. That alone would push YOLO26n on a Pi 5 below 10 FPS. Going down to 480 px saves ~44%.
- **Resolution trade-off.** Unless pufferfish are usually small in frame, a sensible trade is to train at 640 and deploy at 480-640, measuring recall per object-size bucket on the gold set.
- **P2 head.** A P2 head adds a stride-4 feature map, a significant extra cost on CPU, and has no pretrained weights. It is likely not worth it at nano scale on a Pi 5 unless small-fish recall is the dominant failure mode.
- **SAHI-style tiled inference.** This multiplies inference cost by the number of tiles, so it is infeasible at 10 FPS on a Pi 5. *Training* on random crops or zooms is free at inference time and is the more practical option.

### Gaps
- No measured latency of P2-head YOLO variants or SAHI on a Raspberry Pi 5 was found.
- No measured recall gain from P2 heads for fish detection was found.
- No YOLO26 vs YOLO11 small-object (APs) comparison on an underwater dataset was found.

## 6. Overfitting with small data and larger models (observed: YOLO11m generalises worse than YOLO11s)

### Takeaway
Partially documented. Benchmarks on small datasets attribute underperformance of large YOLO variants to overfitting. Underwater studies show nano/small models within ~1-6 mAP points of medium at a fraction of the cost. A counter-example warns that overly tiny models (YOLOX-Nano) can underfit. The project's YOLO11m < YOLO11s observation is consistent with this literature.

### Cited Findings
- **General YOLO benchmark including small datasets:** "the underperformance of larger models like YOLO11x, YOLOv5ux, YOLOv5ul, and YOLOv10l can be attributed to overfitting, especially given the small dataset size". — [YOLO Evolution benchmark, arXiv 2411.00201](https://arxiv.org/pdf/2411.00201) (quote from search snippet; attribution to this paper inferred from the search result, not verified in full text)
- **YOLOv12 sizes, underwater:** nano is only 5.7% less accurate than medium, and runs at 16.48 vs 2.36 FPS on CPU. — [Telematika, YOLOv12 lightweight underwater](https://journal.ithb.ac.id/index.php/telematika/article/view/799)
- **DeepFish + OzFish:** YOLOv12s at 77.6% vs YOLOv11m at 78.3% mAP, with roughly half the size. — [openaccess.ahievran.edu.tr (search summary)](https://openaccess.ahievran.edu.tr/bitstreams/132a4ca7-40f5-430a-a311-b3696523552c/download)
- **Smallest architectures:** a Rhodes University thesis (YOLOv5 vs YOLOv8, underwater fish) found the smallest architectures suited its imbalanced datasets better. — [Rhodes University repository](https://researchrepository.ru.ac.za/handle/20.500.14915/3192)
- **Counter-evidence (underfitting):** YOLOX-Nano's "overly minimal model parameters fail to fit the training data, leading to significant missed detections". — [Aquacultural Engineering 2025](https://www.sciencedirect.com/science/article/abs/pii/S0144860925000226)
- **Counter-evidence (no overfitting with less data):** YOLOv8-s to v11-s trained on 25-100% of coral and fish data improved monotonically with more data, i.e. no overfitting signature at the "s" scale. — [arXiv 2509.12682](https://arxiv.org/abs/2509.12682)
- **Overfitting sign (Ultralytics):** "training loss continues to decrease while validation loss starts to increase". Reduce epochs if this appears early. — [Ultralytics training tips](https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results/)

### Inferences
- **Model size.** With ~725 unique images, the n/s scale is the right regime both statistically and for the Pi-5 latency budget. The YOLO11m result is expected rather than anomalous.
- **Teacher-student option.** Since Pi 5 forces n (or s at reduced resolution), the practical question is n vs s. A larger model (m, or RT-DETR) could still be useful as a teacher for pseudo-labelling or knowledge distillation on a desktop GPU. No direct evidence for this on fish was retrieved.
- **Variance across seeds.** Small-set results are noisy (ICES study SDs up to ±0.02 mAP across 5 repeats). Compare model sizes over multiple seeds before concluding.

### Gaps
- No underwater study was found that directly compares n/s/m of the same YOLO generation on a sub-1,000-image fish dataset with repeated seeds.
- No evidence was found on knowledge distillation from large to nano YOLO for fish detection.
