# Underwater Enhancement, Motion Blur, Tracking and Small-Object Strategies for Pufferfish Detection on Raspberry Pi 5 CPU

Scope: what actually changes detector recall/precision (not visual quality), CPU cost on a Pi 5, and recommended configurations for a moving ROV/boat camera with frequent motion blur. Research date: 2026-10-19. The project's own measurement is that CLAHE preprocessing lowered F1 from 74.6% to 68.2%.

## 1. Does underwater image enhancement (UIE) improve or hurt CNN detector accuracy?

### Takeaway
Across controlled studies from 2023 to 2025, applying a UIE method uniformly as preprocessing does not improve dataset-level detection mAP, and usually lowers it, even when the detector is retrained on the enhanced images. Gains appear only on a minority of very degraded images, or when used as a selective or mixed-domain step. The project's CLAHE result (F1 74.6% to 68.2%) matches the dominant finding. One small fish-tank study reports the opposite, so the effect depends on the dataset.

### Cited Findings
- **Revisiting UIE for detection (J. Imaging 2025)**
  - Setup: 9 UIE methods (ACDC, TEBCF, BayesRet, PCDE, ICSP, AutoEnh, Semi-UIR, USUIR, TUDA), each used to train its own YOLO-NAS-L detector, so training and test domains matched.
  - Dataset-level mAP50-95: no enhanced model beat the original.
    - RUOD (about 14k images, 10 classes): original 0.62 vs enhanced 0.55–0.62. ICSP was worst at 0.55; AutoEnh tied at 0.62.
    - CUPDD (heavily degraded, 414 images): original 0.38 vs enhanced 0.22–0.37.
  - Source: [Revisiting UIE for Object Detection, PMC12842599](https://pmc.ncbi.nlm.nih.gov/articles/PMC12842599/); same results in [arXiv 2411.14626](https://arxiv.org/html/2411.14626v2)
- The same study says enhancement "generally deteriorates the detection performance". Enhancement-quality metrics (UIQM, UCIQE, CCF, Entropy) did not predict detection performance, and the relationship was slightly inverse. — [arXiv 2411.14626](https://arxiv.org/html/2411.14626v2)
- **Per-image analysis in the same study:**
  - Improvements were concentrated in low-to-medium-quality images (quality index < 0.5). High-quality images rarely improved.
  - An oracle "mixed set" that picks the best version (original or enhanced) per image raised per-image mean mAP from 0.68 to 0.77 on RUOD and from 0.41 to 0.64 on CUPDD.
  - Originals were the largest share of that oracle set. Each enhancer contributed only about 4–12% of images.
  - The oracle uses ground truth, so it is an upper bound and cannot be deployed. The authors conclude that enhancement is "most effective when applied selectively rather than uniformly."
  - Source: [PMC12842599](https://pmc.ncbi.nlm.nih.gov/articles/PMC12842599/)
- Enhancement can surface objects that annotators missed. Those then count as false positives, which partly confounds the measured losses. — [PMC12842599](https://pmc.ncbi.nlm.nih.gov/articles/PMC12842599/)
- **"Is Underwater Image Enhancement All Object Detectors Need?" (2023):** applied 18 UIE algorithms (traditional, CNN and GAN), retrained 7 detectors on each output (126 models), and compared them against 7 raw-trained detectors (133 models). It frames itself as the first comprehensive study of this question. — [arXiv 2311.18814](https://arxiv.org/abs/2311.18814)
  - I could not retrieve its numeric tables (the PDF was too large to fetch). Search summaries describe it as highlighting weaknesses of UIE for detection.
- **Remote Sensing 2025 per-image study:**
  - Most enhanced images achieved equal or better per-image detection performance than their originals.
  - Over-enhancement can degrade detection.
  - Mixed-domain training (raw plus enhanced) compensates for enhancement effects.
  - Source: [Understanding the Influence of Image Enhancement on Underwater Object Detection, Remote Sens. 17(2):185](https://doi.org/10.3390/rs17020185)
  - The full text returned HTTP 403, so these points come from the abstract and search summary only.
- **Counter-example (small fish-tank YOLOv11 study):**
  - Accuracy: CLAHE gave mAP@0.5 0.84 vs a 0.69 baseline; gamma plus white balance gave 0.82.
  - Cost: FPS fell from 21.3 to 17 because of CLAHE.
  - Source: [Enhancing Underwater Live Fish Detection with YOLOv11 and Image Enhancement, JTPK](https://journal.ipb.ac.id/jtpk/article/view/66223)
- **Object-type dependence (2025 conference paper):** CLAHE raised detection confidence for opaque objects by about 8%, but reduced detection counts by 44% for translucent objects. — [IITF 2025 proceedings TF071](https://www.iitf.lbtu.lv/conference/proceedings2025/Papers\TF071.pdf)
- **Video flicker:** frame-wise enhancement can introduce flicker and colour drift across frames. Authors of a temporally consistent UIE model argue this lowers detector confidence, inflates false positives and breaks track consistency. — [T-PhISH-Net, Machine Vision and Applications 2026](https://link.springer.com/article/10.1007/s00138-026-01903-z)

### Inferences
- **Why CLAHE hurt here:** the drop from 74.6% to 68.2% F1 is consistent with the literature. Modern YOLO detectors trained on raw underwater images already learn colour and contrast invariance. A fixed enhancement step mainly does three things:
  - amplifies noise, backscatter and particles, which creates false positives;
  - shifts the input distribution away from the training set whenever train and test preprocessing differ;
  - costs CPU time (about 20% FPS loss in the fish-tank study).
- **When enhancement can help:**
  - when the detector was trained on clearer images (for example iNaturalist or diver photos) and deployed in turbid, green or blue water, so there is a train/test domain mismatch;
  - on the most degraded frames;
  - for opaque objects.
  - Pufferfish are opaque. However, the project's own data shows a net loss, so domain mismatch is probably not the dominant error.
- **Augment rather than enhance:** the evidence favours putting enhancement into training augmentation over adding a fixed inference step.
  - Training augmentation: randomly apply CLAHE, gray-world/white balance, hue shift, contrast and haze to a fraction of training images. This is the "mixed-domain training" in the Remote Sensing paper.
  - Inference: run on raw frames. The pipeline gets the robustness benefit at zero inference cost.
- If enhancement is used at inference at all, it should be gated, applied only to frames whose quality index or mean detection confidence is low. This is the selective strategy that the J. Imaging authors suggested but did not test.
- **Cost of learned enhancers on a Pi 5:** learned UIE networks (FUnIE-GAN, Water-Net, Semi-UIR and similar) run a second full-resolution CNN before the detector. That would roughly halve or worse the frame budget, so none is justifiable for a ≥10 FPS target given zero or negative detection gains.
  - This cost estimate is my inference; I found no direct Pi 5 timings.

### Gaps
- I found no study that measured Sea-thru, dark channel prior (DCP), UDCP, Retinex, FUnIE-GAN or Water-Net specifically by detection mAP on a CPU edge device with runtime numbers.
- I could not retrieve the numeric tables of arXiv 2311.18814, or the full text of the Remote Sensing 2025 paper (HTTP 403).
- I found no paper that directly compares "train with enhancement augmentation, test raw" against "enhance at test". The recommendation above combines the mixed-domain finding with reasoning.

## 2. Motion blur: deblurring vs blur augmentation vs shorter exposure

### Takeaway
The strongest quantitative evidence (Sayed et al., CVPR 2021) shows that blur-augmented training, ideally with centred kernels and labels expanded to cover the blur extent, recovers far more AP than deblurring as preprocessing. Deblurring was about 12× slower, inconsistent, and damaged accuracy on sharp images. For a Pi 5, the answer is blur augmentation in training plus shorter exposure at capture. Deblurring at inference is ruled out.

### Cited Findings
All results in this section are from Sayed et al., CVPR 2021 ([arXiv 2011.14448](https://arxiv.org/html/2011.14448)), unless another source is named.
- **Size of the blur problem (Faster R-CNN):**
  - Synthetic blur on COCO: mAP@50 fell from 58.50 (sharp) to 50.95, 15.75 and 4.58 at exposure levels E1, E3 and E5.
  - Real blur datasets:

    | Dataset | Sharp mAP@0.5 | Blurry mAP@0.5 |
    |---|---|---|
    | GOPRO | 35.85 | 19.64 |
    | RealBlur | 42.39 | 29.00 |
    | REDS | 42.02 | 24.38 |
- **Remedies at the most severe blur level (E5, standard labels, baseline 4.58 mAP@50):**

  | Remedy | mAP@50 |
  |---|---|
  | Deblur then original detector | 6.26 |
  | AugMix | 8.21 |
  | Standard blur augmentation | 31.37 |
  | Specialised by exposure (best) | 35.93 |

  - With expanded labels, the best method reached 43.78.
  - Centring the blur kernel added about 5.7 points (25.66 non-centred vs 31.37 centred).
- **Deblurring as preprocessing:**
  - It reduced clean COCO accuracy from 58.50 to 55.50.
  - It collapsed sharp GOPRO accuracy from 35.85 to 10.58.
  - On real blur it was inconsistent: GOPRO blurry 19.64 → 2.24, RealBlur flat, REDS +7.5.
  - It ran about 12× slower.
  - The authors conclude deblurring "only partly improves object detection".
- **Real-world blur with the best augmentation recipe:** gains of about 7–9 mAP points.

  | Dataset | Blurry baseline | With best recipe |
  |---|---|---|
  | GOPRO | 19.64 | 28.15 |
  | RealBlur | 29.00 | 36.37 |
  | REDS | 24.38 | 32.55 |
- **Limits of COCO-C as a benchmark:** COCO-C applies ImageNet-C-style corruptions, including straight-line motion blur. Its blur types (Gaussian, defocus, glass, camera shake, zoom) are not caused by object motion, so it may understate real failure modes. — [ECCV 2024, Retrieval Robust to Object Motion Blur](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11229.pdf)
- **Detection-driven deblurring:** Deblur-YOLO integrates blind deblurring into a one-stage real-time detector, which shows there is interest in end-to-end approaches. — [Deblur-YOLO, ResearchGate](https://www.researchgate.net/publication/354797924_Deblur-YOLO_Real-Time_Object_Detection_with_Efficient_Blind_Motion_Deblurring)

### Inferences
- **Training recipe:**
  - Add a motion-blur augmentation (for example Albumentations MotionBlur, linear kernels of random angle, length about 3–25 px at 640 input) to roughly 20–30% of training images.
  - Keep the blur kernel centred on the object.
  - Consider slightly loosening box tightness on heavily blurred samples, which approximates the "expanded labels" idea.
- **Capture settings:** a shorter exposure on the Pi camera (for example a fixed shutter of about 1/250–1/500 s with analog gain raised) removes blur at its source at zero CPU cost.
  - The trade-off is sensor noise, which blur and noise augmentation in training can cover.
  - I found no underwater-specific study quantifying this trade-off.
- **Deblurring at inference** (DeblurGAN or similar) is not viable on a Pi 5 at ≥10 FPS. It costs about 12× compute, and the detection gain is inconsistent and can be negative.
- **Frame selection:** a cheap blur score, such as variance of the Laplacian on a downscaled frame (about 1 ms on CPU), can mark frames as unreliable. The tracker then coasts on its prediction instead of starting new tracks from those frames. This is an engineering suggestion, not a tested result.

### Gaps
- I found no underwater- or fish-specific benchmark of detector robustness to motion blur, and no Pi-specific deblurring timings.
- I found no quantitative study of exposure time vs detection recall for underwater cameras.

## 3. Multi-object tracking for false-alarm reduction and counting on edge CPUs

### Takeaway
SORT-family trackers that use only motion (SORT, ByteTrack, OC-SORT) cost under about 2 ms per frame on a desktop CPU core, which is negligible next to detector inference on a Pi 5. ReID-based trackers (DeepSORT, BoT-SORT with ReID) add real CPU cost. On fish data, ByteTrack and OC-SORT are close in IDF1, and hybrids (Byte-OCSORT) do best for counting. Temporal confirmation through the tracker's track-birth rules is the standard false-positive filter. I found no paper that quantifies an "N of M frames" rule specifically for fish.

### Cited Findings
- **Tracker speed on CPU:**
  - OC-SORT runs at 793 FPS on an Intel i9-9980XE CPU, given detections. — [OC-SORT, arXiv 2203.14360](https://arxiv.org/pdf/2203.14360); "700+ FPS on a single CPU" in the [CVPR 2023 poster](https://cvpr.thecvf.com/virtual/2023/poster/22117)
  - An independent single-core measurement (AMD Ryzen 9 3900) of the tracking stage only:

    | Benchmark | ByteTrack | OC-SORT |
    |---|---|---|
    | JRDB-panorama | 575 FPS | 380 FPS |
    | OmniSmall | 1459 FPS | 670 FPS |

    So ByteTrack was faster than OC-SORT in this setup. — [Re-engineering SORT-based algorithms for low-cost small object tracking, arXiv 2609.07547](https://arxiv.org/pdf/2609.07547)
- **Fish tracking accuracy:**
  - Real recirculating aquaculture system (RAS) footage: YOLOv8+ByteTrack reached IDF1 59.96% vs 58.76% for YOLOv8+OC-SORT. OC-SORT had better ID precision but lower recall. — [FishTracker, CMC/ScienceDirect 2025](https://www.sciencedirect.com/org/science/article/pii/S1546221825012676)
  - FishTracker adds an adaptive Kalman filter on top of ByteTrack; its detector runs at 21.2 FPS (workstation hardware). — same source
  - Underwater multi-fish dataset: ByteTrack IDF1 52.2 vs OC-SORT 49.7. — [Enhanced CenterTrack, Animals 2026](https://doi.org/10.3390/ani16020156)
  - Fish counting: a Byte-OCSORT hybrid (OC-SORT plus ByteTrack's two-stage low-score matching) reported the best MOTA (72.3) and IDF1 (69.4) against SORT, ByteTrack and OC-SORT. — [Improved YOLOv8 and OC-SORT Framework for Fish Counting, JMSE 13(6):1016](https://www.mdpi.com/2077-1312/13/6/1016)
    - The full text returned HTTP 403, so these numbers come from the search summary.
- **Low-confidence matching:** the SU-T baseline in the MFT25 benchmark uses a second association stage that matches leftover tracks to low-confidence detections. The authors report this recovers occluded fish while filtering false positives caused by turbidity and reflections. — [When Trackers Date Fish, arXiv 2507.06400](https://arxiv.org/pdf/2507.06400)
- **Gated track memory:** AquaMambaTrack writes observations into a track's identity memory only after stage, geometry, appearance and crowd-reliability checks. This is a form of confirmation gating. — [AquaMambaTrack, JMSE 2026](https://doi.org/10.3390/jmse14171625)
- **Track ageing:** Enhanced CenterTrack lets track confidence decay and purges tracks that stay unmatched past a set age, which limits false-positive drift. — [Animals 2026](https://doi.org/10.3390/ani16020156)
- **Ultralytics tracking:**
  - Supported trackers: BoT-SORT, ByteTrack, OC-SORT, Deep OC-SORT, FastTracker and TrackTrack (the default).
  - Key parameters:
    - `track_high_thresh` / `track_low_thresh`: confidence levels for the two-stage association;
    - `new_track_thresh`: confidence needed to start a track;
    - `track_buffer`: frames a lost track is kept;
    - `match_thresh`;
    - `gmc_method`: camera-motion compensation, with options sparseOptFlow (BoT-SORT default), orb, sift, ecc or none.
  - ByteTrack, OC-SORT and FastTracker do not use camera-motion compensation (CMC). Setting `gmc_method: none` saves "a few ms/frame".
  - ReID is off by default "to minimize overhead".
  - Source: [Ultralytics Track mode docs](https://docs.ultralytics.com/modes/track/)
- **Video detectors that reuse a tracker:** Yao et al. (ECCV 2020) run the detector on sparse keyframes and lightweight trackers in between. They report state-of-the-art ImageNet VID results in real time on CPU. — [ECCV 2020, Object-level Temporal Aggregation](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590154.pdf)
- **ID switches and counting:** identity switches cause line-crossing counters to double count. — [rizwanai tracker comparison](https://www.rizwanai.com/blog/yolo-object-tracker-comparison-botsort-bytetrack-ocsort-deepocsort-fasttrack-tracktrack) (blog, lower reliability)
- **No Pi benchmark found:** I found no peer-reviewed Raspberry Pi benchmark of ByteTrack or OC-SORT on fish footage.

### Inferences
- **Expected tracker cost on a Pi 5:**
  - A Cortex-A76 core is several times slower than a desktop core, so ByteTrack or OC-SORT should take roughly 1–5 ms per frame for a handful of fish. That is under 5% of a roughly 100 ms detector frame budget (estimate).
  - BoT-SORT with sparseOptFlow camera-motion compensation adds optical-flow cost of a few ms or more at full resolution. ReID adds a CNN per detection and should be avoided on a Pi.
- **Recommended confirmation rule for false alarms:**
  - Report a pufferfish only when a track has been matched in at least 3 of the last 5 frames (about 0.3–0.5 s at 10 FPS). In ByteTrack terms, this corresponds to a minimum number of consecutive hits before a track is activated, plus `new_track_thresh` set higher than `track_high_thresh`.
  - Allow `track_buffer` of about 15–30 frames so a blurred or occluded fish is not re-counted.
  - Exploit ByteTrack's low-score second stage: keep the detector confidence threshold low (for example 0.1), so faint, distant or blurred pufferfish keep existing tracks alive. Only confirmed tracks raise alerts.
- **Choosing a tracker for a moving camera:**
  - OC-SORT's observation-centric re-update handles non-linear motion and lost frames better.
  - BoT-SORT's camera-motion compensation explicitly corrects ego-motion but costs optical flow.
  - At the target of about 10 FPS with large inter-frame displacement, OC-SORT or ByteTrack with a permissive `match_thresh` is the cheapest starting point. Benchmark BoT-SORT with `gmc_method: sparseOptFlow` on the project's own ROV clips before adopting it.
- **Counting:** count unique confirmed track IDs, or line or region crossings, rather than per-frame detections. Use the hybrid two-stage association (Byte-OCSORT-style) if ID switches inflate counts.
- **Evaluation:** report precision, recall and F1 both per frame and after temporal confirmation. The false-positive reduction claimed for temporal filtering has to be measured on project videos, because I found no published fish N-of-M statistic.

### Gaps
- I found no published quantification of how much an N-of-M temporal confirmation rule reduces false positives for underwater fish detection.
- I found no Pi 5 ARM timings for ByteTrack, OC-SORT, BoT-SORT or Norfair. The Norfair literature was not searched in depth.
- Default numeric values of the Ultralytics tracker YAML parameters are not given on the docs page; check the repo configs.

## 4. Small objects: SAHI / tiling vs higher resolution vs ROI zoom on a CPU budget

### Takeaway
SAHI raises small-object AP substantially. Slicing at inference alone added about 5–7 AP, and with slice-aware fine-tuning about 12–14.5 AP. However, compute scales roughly linearly with the number of tiles. A nano YOLO already runs at only about 7.5 FPS at 640 on a Pi 5 CPU, so full-frame SAHI is incompatible with a ≥10 FPS target. Practical options are a modest resolution increase, adaptive or ROI-gated tiling at low frequency, or tracker-guided crops.

### Cited Findings
- **SAHI (Akyon et al. 2022):**
  - Slicing at inference raised AP by about 5–7% across FCOS, VFNet and TOOD on VisDrone and xView.
  - Slicing-aided fine-tuning brought cumulative gains to about 12–14.5%.
  - Overlapping slices mainly help small and medium objects and slightly hurt large objects, because slices produce false positives on large boxes. Full-image inference merged with the slices helps large objects.
  - Source: [SAHI, arXiv 2202.06934](https://arxiv.org/pdf/2202.06934)
- **Tiling cost:** inference cost rises with the number of tiles, which is especially wasteful on sparse or mostly-background scenes. ROI-gated SAHI proposes slicing only content-bearing regions. — [ROI-Gated SAHI, arXiv 2608.23923](https://arxiv.org/pdf/2608.23923)
- **Adaptive slicing (ASAHI, April 2026):** fixed slice sizes cause redundant computation, so ASAHI picks the slice count from image resolution. It beat SAHI on VisDrone by modest margins. — [ASAHI, arXiv 2604.19233](https://arxiv.org/pdf/2604.19233)
- **Pi 5 baseline cost:**
  - Setup: ONNX Runtime CPU, active cooling at 2.4 GHz, models trained at 640.

    | Model | ms per image | FPS |
    |---|---|---|
    | YOLO26n | 133.2 | 7.51 |
    | YOLOv5nu | — | 7.19 |
    | YOLO11n | — | 6.71 |
    | YOLOv8n | — | 6.51 |
    | YOLO26s | 345.2 | 2.90 |
    | Medium models | — | 1.04–1.25 |

  - Without active cooling, the Pi 5 throttled within minutes, with peaks above 80 °C.
  - Source: [Does YOLO26 Truly Offer Advantages… Aquaculture, arXiv 2607.09835](https://arxiv.org/pdf/2607.09835)

### Inferences
- **Frame-rate budget on a Pi 5:**
  - At 640 input, nano models give about 6.5–7.5 FPS with ONNX Runtime. Reaching ≥10 FPS already requires a smaller input (for example 512 or 416) and/or a faster runtime (NCNN or OpenVINO-ARM, INT8). I did not verify those runtimes here; another workstream should cover them.
  - Latency scales roughly with pixel count:
    - 512² ≈ 0.64× and 416² ≈ 0.42× the pixels of 640², i.e. about 10–15 FPS for YOLO26n (estimate).
    - SAHI with 4 tiles at 640 ≈ 4× the cost, about 2 FPS. That is unusable in real time.
- **Recommended small-object strategy within budget:**
  1. Train at the deployment resolution and include small and distant pufferfish in training, using scale augmentation or copy-paste of small fish. This costs nothing at inference.
  2. If distant fish matter, run the full frame at about 416–512 every frame. Every N frames (for example once a second), or when the tracker has a low-confidence or lost track, run 1–2 high-resolution crops (ROI zoom) around candidate regions. This is a tracker-guided ROI-gated SAHI.
  3. Prefer a single moderate resolution increase over many tiles when the objects are only moderately small.
- **Practical distance limit:** distant fish in turbid water lose contrast, not just pixels. Below roughly 10–15 px object size, neither tiling nor resolution helps much without a better camera or optics. This is a heuristic; I found no underwater-specific measurement.

### Gaps
- I found no Pi 5 measurement of SAHI or tiled inference.
- I found no underwater or fish-specific SAHI study.
- I found no study comparing higher input resolution against tiling at equal CPU cost.

## 5. NMS settings, duplicate boxes, and NMS-free detectors (YOLO26, YOLOv10)

### Takeaway
NMS-free detectors (YOLOv10, YOLO26) remove duplicate suppression from post-processing through one-to-one label assignment. On a Pi 5 this gave YOLO26n a 4.5–15% speed edge over other nano models, much less than the "up to 43%" vendor claim, which was measured on a Xeon. Accuracy on a fish dataset was within about 1 mAP50 point of the NMS-based models. For NMS-based models, the IoU threshold controls duplicate boxes. I found no fish-specific study of the optimal threshold.

### Cited Findings
- **YOLO26 on the Pi 5 (aquaculture benchmark):**
  - YOLO26n was the fastest nano model (7.51 FPS), 11.9% faster than YOLO11n. The authors attribute the gain mainly to removing NMS and DFL post-processing.
  - The gain was well below Ultralytics' "up to 43%" CPU ONNX claim, which was measured on Intel Xeon.
  - Post-processing time:
    - YOLO26n: 0.179 ms (18% of inference plus post-processing);
    - YOLOv8n: 1.805 ms (67.8%);
    - YOLO11n: 0.702 ms.
  - The paper describes YOLO26's design as "handles duplicate suppression within the network".
  - Source: [arXiv 2607.09835](https://arxiv.org/pdf/2607.09835)
- **Same benchmark, accuracy and data needs:**
  - All 12 models had mAP50 within 1.04 points on the full fish dataset.
  - YOLO26 needed about 2.5× more training images than YOLOv8 to pass 90% mAP50 ("reduced training efficacy and higher data requirements").
  - On the A100 GPU, YOLO26 had the highest end-to-end latency despite its tiny post-processing, so the speed benefit is specific to CPUs.
  - Source: [arXiv 2607.09835](https://arxiv.org/pdf/2607.09835)
- **Vendor framing:** Ultralytics positions YOLO26 and YOLOv10 as end-to-end, NMS-free models. It recommends YOLO26 for Raspberry Pi-class CPUs and says that removing NMS reduces latency variability. — [Ultralytics YOLO26 vs YOLOv10](https://docs.ultralytics.com/compare/yolo26-vs-yolov10) (vendor source)
- **YOLOv10:** introduced consistent dual assignments (one-to-many for training supervision and one-to-one for inference) to remove NMS. — [YOLOv10, arXiv 2405.14458](https://arxiv.org/abs/2405.14458)
  - I did not fetch this page during the session; the claim is cited from the known paper.

### Inferences
- **Fixing duplicate boxes:**
  - The project has previously noted a duplicate-box issue. With an NMS-based model (YOLOv8 or YOLO11), lower the NMS IoU threshold from the Ultralytics default of 0.7 to about 0.45–0.5. For a single class, also consider class-agnostic NMS.
  - Duplicates on one fish usually overlap heavily, while two real pufferfish rarely overlap above 0.5 IoU, so the risk to recall is small. This should be validated against the gold_v1 test set.
- **Switching to YOLO26n:** removes the NMS IoU tuning knob and gives about a 5–15% speed gain on a Pi 5. Two cautions:
  - The one-to-one head can still emit near-duplicates at low confidence thresholds. A tracker naturally merges these, since ByteTrack's association handles them.
  - YOLO26 may need more training data than YOLOv8 to reach the same accuracy (aquaculture benchmark).
- **Interaction with the tracker:** duplicate boxes can spawn spurious extra tracks and inflate counts. Fix duplicates (NMS IoU, or the NMS-free head) before tuning the tracker.

### Gaps
- I found no published study of NMS IoU threshold vs duplicate rate for fish detection.
- I found no direct measurement of the residual duplicate rate of YOLO26 or YOLOv10 vs tuned NMS on fish data.

## 6. Video-based detection (temporal feature aggregation): anything light enough for a Pi 5?

### Takeaway
True temporal-feature-aggregation detectors (flow-guided aggregation, ConvLSTM detectors) were designed for mobile, but date from 2017–2020, are not supported by current YOLO toolchains, and need custom training. For a Pi 5 the practical "video method" is tracking-by-detection with keyframe scheduling: a detector on some frames and tracker propagation in between, plus temporal confirmation.

### Cited Findings
- **Bottleneck-LSTM mobile video detector (Liu & Zhu 2017/2018):** an SSD-style detector with convolutional LSTM layers reached "up to 15 FPS on a mobile CPU", with accuracy close to heavier single-frame models on ImageNet VID 2015. — [Mobile Video Object Detection with Temporally-Aware Feature Maps, arXiv 1711.06368](https://arxiv.org/pdf/1711.06368)
- **Light Flow plus flow-guided GRU (2018):** propagates features cheaply on non-key frames and aggregates on sparse keyframes. It was benchmarked with TensorFlow Lite on a single ARM Cortex-A72 core. — [Towards High Performance Video Object Detection for Mobiles, arXiv 1804.05830](https://arxiv.org/pdf/1804.05830)
- **Object-level temporal aggregation (Yao et al., ECCV 2020):** a detector on sparse keyframes with lightweight trackers in between. Real-time on CPU with state-of-the-art ImageNet VID results. — [ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590154.pdf)

### Inferences
- **Recommended approach:**
  - Do not adopt feature-aggregation video detectors; the tooling and training cost is high for a single-class TÜBİTAK project.
  - Instead, run detector plus ByteTrack/OC-SORT, optionally with keyframe skipping:
    - run the detector every frame when ≥10 FPS is achievable;
    - otherwise detect every 2nd frame and let the Kalman prediction fill in. This roughly doubles effective FPS at some recall cost on fast motion.
  - The Pi 5 Cortex-A76 is faster than the Cortex-A72 used in the 2018 mobile benchmarks, so the 2018 numbers are conservative.

### Gaps
- I found no 2022+ lightweight video detector with Pi 5 or ARM benchmarks, and no underwater video-detection method evaluated on edge CPUs.
- I could not verify the detailed FPS and mAP tables of the 2018 Light Flow paper.

### Inferences: overall recommended Pi 5 configuration (synthesis of all sections above; my inference, not a tested result)

1. **No inference-time enhancement.** Feed raw frames to the detector. Move CLAHE, white balance, haze and colour-shift into training augmentation (about 20–30% probability). If anything is used at runtime, use only the cheap gray-world white balance, and only after an ablation on gold_v1 shows a gain.
2. **Motion blur:** add motion-blur augmentation in training (linear kernels, random angle, a range of lengths). Use a short fixed exposure on the camera. Do not deblur at inference.
3. **Detector:** a nano model (YOLO26n or YOLO11n/v8n) at about 416–512 input, with an optimised runtime, under active cooling. If an NMS-based model is used, set the NMS IoU to about 0.5 and use class-agnostic NMS.
4. **Tracker:** ByteTrack or OC-SORT, with no ReID. Use a low detection threshold (about 0.1–0.25) for the second association stage and a higher `new_track_thresh` (about 0.5). Confirm a track as a pufferfish after 3 matches in 5 frames, and keep lost tracks for about 15–30 frames. Count unique confirmed track IDs.
5. **Small or distant fish:** use scale and copy-paste augmentation in training. Optionally run occasional ROI zoom crops around low-confidence or lost tracks instead of full-frame SAHI.
6. **Measure:** per-frame and track-level precision, recall and F1 on project video clips, plus end-to-end FPS with the tracker included.
