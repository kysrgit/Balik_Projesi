# CPU-only YOLO inference on Raspberry Pi 5 (Cortex-A76): runtimes, precision, settings (2024-2026)

Scope note: Numbers are Raspberry Pi 5 (BCM2712, 4x Cortex-A76 @ 2.4 GHz) unless explicitly flagged. The "user measured" figures come from this project's own tests (supplied in the assignment), not from web sources: ORT FP16 gave no speedup (slower for YOLO11m); ORT static INT8 mixed precision gave 2.1x; NCNN with fp16 arithmetic gave YOLO11n 70 ms and YOLO11s 162 ms end-to-end at 640.

## 1. Runtime comparison on Pi 5 (NCNN, ORT, LiteRT/TFLite, OpenVINO, MNN, ExecuTorch, PyTorch)

### Takeaway
NCNN is the fastest CPU runtime for YOLO on Pi 5 in every Ultralytics table from 2024 to 2026. It is about 2x faster than ONNX Runtime and 4-5x faster than PyTorch. MNN and (recent) OpenVINO come second, and ONNX, LiteRT and ExecuTorch form a third tier. Expect roughly 67-94 ms (11-15 FPS) for a nano model at 640 FP32-export, and about 160-200 ms (5-6 FPS) for a small model.

### Cited Findings
- **Ultralytics YOLO26n, Pi 5, all formats, 640, Ultralytics 8.4.108.** Inference only, excluding pre/post-processing. Columns are size / mAP50-95 / ms per image. [Ultralytics Raspberry Pi guide](https://docs.ultralytics.com/guides/raspberry-pi/)
  - PyTorch: 5.3 MB / 0.4760 / 299.09 ms
  - TorchScript: 9.8 / 0.4734 / 353.20
  - ONNX: 9.5 / 0.4734 / 125.99
  - OpenVINO: 9.6 / 0.4734 / 104.55
  - LiteRT: 9.8 / 0.4730 / 123.30
  - MNN: 9.4 / 0.4749 / 91.87
  - **NCNN: 9.4 / 0.4784 / 67.03**
  - ExecuTorch: 9.4 / 0.4772 / 144.83
  - The guide states that "NCNN delivers the best inference performance on Raspberry Pi devices". The table does not name its dataset: the reproduce example uses coco128, and the guide recommends coco.yaml for reliable numbers. [Ultralytics Raspberry Pi guide](https://docs.ultralytics.com/guides/raspberry-pi/)
- **Earlier YOLO26n table (Ultralytics 8.4.1, French mirror of the same guide).** NCNN ~67.69 ms (mAP 0.4805), OpenVINO ~70.7 ms, MNN ~90.9 ms, ONNX ~130 ms, PyTorch ~302 ms. These values come from a search-result snippet, not a full fetch. [Ultralytics guide (FR)](https://docs.ultralytics.com/fr/guides/raspberry-pi)
- **YOLO26 vs YOLO11 on Pi 5 (ONNX, 640, Ultralytics 8.4.14).** YOLO26n: 40.1 mAP50-95 (COCO), 128.42 ms. YOLO26s: 47.8 mAP, 352.84 ms. The guide quotes YOLO11n at 6.79 FPS vs YOLO26n at 7.79 FPS (mAP 39.5 vs 40.1). [Ultralytics Raspberry Pi guide](https://docs.ultralytics.com/guides/raspberry-pi/)
- **Older YOLO11 Pi 5 tables** (2024-era guide; mirror; Ultralytics version not stated; mAP values ~0.61/0.75 imply a tiny val set such as coco8/coco128). [Ultralytics guide mirror (torch-book)](https://torch-book.readthedocs.io/zh-cn/old/ecosystem/ultralytics/ref/guides/raspberry-pi.html)
  - YOLO11n ms/im: PyTorch 524.8 | TorchScript 666.9 | ONNX 181.8 | OpenVINO 530.2 | TF SavedModel 406.0 | TF GraphDef 473.6 | TF Lite 324.2 | PaddlePaddle 644.3 | **NCNN 93.9**
  - YOLO11s ms/im: PyTorch 1226.4 | TorchScript 1508.0 | ONNX 415.2 | OpenVINO 1167.1 | TF SavedModel 776.1 | TF GraphDef 1014.4 | TF Lite 845.9 | PaddlePaddle 1567.8 | **NCNN 197.4**
- Another version of the Ultralytics table lists YOLO11n NCNN on Pi 5 at 88.026 ms/im (640, FP32). This comes from a search summary of docs.ultralytics.com and is not independently fetched. [Ultralytics Raspberry Pi guide](https://docs.ultralytics.com/guides/raspberry-pi)
- **Independent aquaculture benchmark (fish mortality detection, 2026).** It covered 12 variants (YOLO26, YOLOv5u, YOLOv8, YOLO11; n/s/m) on CPU-only Pi 5. YOLO26n was the fastest at 7.51 FPS. YOLO26s ran at 345.2 ms (2.90 FPS), which is consistent with PyTorch/ONNX-class runtimes rather than NCNN. YOLOv5mu beat the other medium models on CPU. mAP50 across all models differed by only 1.04 pp on the full dataset. YOLOv8 reached 90% mAP50 with 400 training images, while YOLO26 n/s needed ~1,000. Ranjan et al., MDPI AI 2026, 7, 354. [arXiv 2607.09835](https://arxiv.org/abs/2607.09835)
- **Arm's ExecuTorch + XNNPACK + KleidiAI YOLOv8s on Pi 5 (Cortex-A76 @ 2.4 GHz, Debian 13, ExecuTorch 1.1.0, 640).**
  - Baseline: 1018.84 ms p50, described as "PyTorch eager".
  - Arm-optimized INT8: 457.03 ms p50 (2.19 FPS), a 2.23x speedup.
  - Peak memory: 141 -> 90 MB.
  - [Arm/yolov8s-int8-xnnpack-executorch (Hugging Face)](https://huggingface.co/Arm/yolov8s-int8-xnnpack-executorch)
- **Related Arm records.** YOLOv5s INT8: 618.3 ms baseline -> 258.1 ms optimized (end-to-end p50). YOLOv9s INT8: 463.9 ms p50. These come from search snippets of Arm's Hugging Face model cards. [Arm yolov5s card](https://huggingface.co/Arm/yolov5s-int8-xnnpack-executorch); [Arm yolov9s card](https://huggingface.co/Arm/yolov9s-int8-xnnpack-executorch)
- **ncnn's own benchmark (official README, 4 threads, powersave=0, 10 loops).** Pi 5 vs Pi 4 avg ms:
  - mobilenet_v2: 10.52 vs 61.88
  - resnet18: 19.96 vs 123.12
  - mobilenet_v3: 7.35 vs 47.53
  - Other Pi 5 rows: yolov4-tiny 46.29, nanodet_m 11.18, yolo-fastestv2 4.80, resnet50 46.21
  - This makes Pi 5 roughly 6x faster than Pi 4 in ncnn. No YOLOv8/YOLO11 rows are in the README.
  - [ncnn benchmark README](https://github.com/Tencent/ncnn/blob/master/benchmark/README.md)
- **ncnn Vulkan on the Pi 5 VideoCore VII GPU is much slower than its CPU for FP models.** mobilenet_v2: 121.46 ms on GPU vs 10.52 ms on CPU. yolov4-tiny: 674 ms vs 46 ms. Overclocking to 2.9 GHz CPU / 1.1 GHz GPU still gives 589 ms for yolov4-tiny on GPU. [ncnn benchmark README](https://github.com/Tencent/ncnn/blob/master/benchmark/README.md)
- **ONNX Runtime XNNPACK EP.** Official docs list prebuilt XNNPACK packages for Android (Maven) and iOS (CocoaPods). For Linux they only describe building ORT from source with `--use_xnnpack`. No source found that confirms the PyPI aarch64 `onnxruntime` wheel includes XNNPACK. [ORT XNNPACK EP docs](https://onnxruntime.ai/docs/execution-providers/Xnnpack-ExecutionProvider.html)

### Inferences
- For this project the ranking is clear: NCNN > MNN ≈ OpenVINO (2026 builds) > LiteRT ≈ ONNX Runtime > ExecuTorch > PyTorch/TorchScript.
- OpenVINO on ARM improved a lot between the YOLO11-era table (530 ms YOLO11n, slower than ONNX) and the 2026 YOLO26 tables (70-105 ms). It is now worth a re-test, but it still does not beat NCNN.
- The user's measured NCNN numbers are consistent with or better than Ultralytics' tables, even though they are end-to-end while Ultralytics' are inference-only. YOLO11n 70 ms compares with Ultralytics' 88-94 ms, and YOLO11s 162 ms with 197 ms. This suggests the user's NCNN build/settings are already near the state of the art for Pi 5 CPU.
- YOLO26n in NCNN (~67 ms inference) is a few ms faster than YOLO11n and is NMS-free, which removes NMS post-processing cost. Even so, the aquaculture paper suggests YOLO26 needs more training data to match YOLOv8 accuracy. This is relevant for a small fish dataset.
- Arm's "baseline" (1019 ms PyTorch eager for YOLOv8s) is far slower than Ultralytics' PyTorch figures (YOLO11s 1226 ms is comparable). Its INT8 ExecuTorch result (457 ms) is still ~2.3x slower than NCNN FP for an s-model (~160-200 ms). INT8 via ExecuTorch is therefore not competitive with NCNN on Pi 5 today.

### Gaps
- No full Pi 5 per-format table for YOLO11s/YOLO26s from 2026 (the 2026 Ultralytics page shows only one set of values per format; the column-to-variant mapping is unclear).
- No independent Q-engineering or Jeff Geerling Pi 5 YOLO11/YOLO26 CPU benchmark was retrieved. learnopencv's "YOLO11 on Raspberry Pi" article returned HTTP 403.
- Arm NN / Arm Compute Library direct YOLO benchmarks on Pi 5: none found.
- MNN-specific docs and benchmarks beyond Ultralytics' row: not retrieved.
- Whether `pip install onnxruntime` on aarch64 exposes `XnnpackExecutionProvider` is unconfirmed. Check `onnxruntime.get_available_providers()` on the Pi.

## 2. Precision on Cortex-A76: FP32 vs FP16 vs INT8 (which runtimes exploit asimdhp / asimddp)

### Takeaway
On A76, the big win comes from FP16 arithmetic, and only NCNN uses it automatically. LiteRT/XNNPACK can use it, but only with FP16-flagged models or a force flag. ONNX Runtime's CPU EP mostly up-casts FP16 to FP32, which explains the user's "FP16 no speedup / slower" result. INT8 helps less than one might expect on A76 once FP16 arithmetic is in use: ncnn shows only ~1.1-1.2x over its FP16 path at 4 threads, and INT8 is slower at 1 thread. Large INT8 gains (2x+) appear only relative to an FP32 baseline, as in the user's ORT result and Arm's ExecuTorch result.

### Cited Findings
- **NCNN computes FP32 exports in FP16 by default** on CPUs that support it. The Ultralytics export flag `quantize=16` (replacing the deprecated `half`) only halves weight storage. The Ultralytics NCNN page documents no int8 export option. [Ultralytics NCNN integration](https://docs.ultralytics.com/integrations/ncnn/)
- **ncnn Pi 5 benchmark (4 threads):**
  - resnet18: 19.96 ms default (fp16-arith) vs 16.74 ms int8 → 1.19x
  - resnet50: 46.21 vs 40.75 → 1.13x
  - At 1 thread, INT8 is slower: resnet18 29.68 vs int8 36.42; resnet50 89.43 vs 80.16 (int8 slightly faster there).
  - On Pi 4 (A72, no FP16 arithmetic), resnet18 int8 gives a larger gain: 123.12 → 83.92 ms (1.47x).
  - [ncnn benchmark README](https://github.com/Tencent/ncnn/blob/master/benchmark/README.md)
- **ONNX Runtime CPU and FP16.** An ORT maintainer stated (Feb 2024) that the CPU "cannot run fp16 in most computation operators, and need to convert to fp32", and that an "fp32 model shall be faster than fp16 model in CPU". [ORT issue #19437](https://github.com/microsoft/onnxruntime/issues/19437)
- ORT's arm64 FP16 kernels (NEON fp16/fp32 conversion, hgemm for specific ops, 8-bit NEON paths) were still landing in 2025-2026 commits, so behaviour depends on version. The default fallback is to "convert the data to fp32 and calculate". [ORT commit 1942e40e](https://app.semanticdiff.com/gh/microsoft/onnxruntime/commit/1942e40e05ade03a3aeecfa2435f03ccf897415e); [ORT commit c1f38c03](https://app.semanticdiff.com/gh/microsoft/onnxruntime/commit/c1f38c0379ec0928bd6f3626c8b08f1348afb64b)
- **LiteRT/TFLite XNNPACK FP16.** Supported on ARMv8.2-FP16 CPUs but not on by default. The model needs FP16 weights plus `reduced_precision_support` metadata (`_experimental_supported_accumulation_type = tf.float16`), or you force it with `TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16`. Google reported "close to 2X" single-threaded speedups on phones. Raspberry Pi / Linux aarch64 was not mentioned. [TensorFlow blog, Nov 2023](https://blog.tensorflow.org/2023/11/half-precision-inference-doubles-on-device-inference-performance.html)
- **OpenVINO CPU.** Docs list f16 as a supported inference precision on Arm. A 2024 commit (#22992) enabled both f16 and f32 kernels for aarch64 with a runtime FP16-support check, replacing an unconditional f16 default under a compile flag. The generic docs say the default CPU primitive precision is f32. Whether f16 is the default on A76 in current releases is unconfirmed. [OpenVINO commit #22992](https://cdn09022024.gitlink.org.cn/natas_hw/openvino/commit/cda5a0239f5f661dc729cc2958fd61884c04082a); [OpenVINO CPU device docs 2024](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html)
- **ExecuTorch + XNNPACK + KleidiAI INT8 (symmetric per-channel PTQ) on Pi 5.** YOLOv8s 2.23x vs PyTorch eager FP32 baseline. [Arm HF card](https://huggingface.co/Arm/yolov8s-int8-xnnpack-executorch)

### Inferences
- The user's ORT results are explained by the sources above. ORT FP16 adds Cast overhead without FP16 math, so no speedup (or a slowdown). ORT INT8 (QLinearConv/QDQ on MLAS, which can use the A76 dot-product instructions) beats an FP32 baseline by ~2x.
- NCNN already runs at FP16-arithmetic speed by default. In ncnn's own tests INT8 adds only ~10-20% at 4 threads. Since the user's ORT INT8 (2.1x over ORT FP32) still lands roughly at NCNN-FP16 level, NCNN-FP16 should remain the default. NCNN INT8 is only worth trying for a possible 10-20% gain, after checking the accuracy cost.
- For LiteRT, an FP16 path exists in principle (XNNPACK FP16 on asimdhp), but Ultralytics' TFLite/LiteRT export does not appear to set the reduced-precision metadata. This is likely why LiteRT trails NCNN by ~2x in the tables. This is an inference, not verified.

### Gaps
- No published YOLOv8/11/26 NCNN INT8 vs FP16 latency/mAP comparison on Pi 5 was found. The ncnn data is for classification/legacy detectors (resnet, mobilenet-ssd).
- No source that explicitly states which ORT MLAS kernels use the `sdot`/`udot` (asimddp) instructions on A76. Plausible but unverified.
- MNN FP16/INT8 behaviour on A76: not researched (no source retrieved).

## 3. INT8 quantization best practice for YOLO detection heads

### Takeaway
Use static PTQ with per-channel symmetric weights and a few hundred to ~2,000 calibration images. Keep the first conv, the final head prediction convs, DFL decode, sigmoid and concat in FP32. With these exclusions, published results show ≤0.5 mAP50-95 loss. Naïve full-INT8 including the head loses 2-5+ points. QAT is rarely needed for n/s models if the head is excluded.

### Cited Findings
- **Arm's Pi 5 YOLOv8s INT8 recipe:**
  - PT2E static PTQ, symmetric per-channel weights.
  - Calibration on a stratified subset of 2,000 COCO images.
  - Kept in FP32: first conv, detection-head prediction convs (cv2/cv3 final conv), DFL decode, sigmoid and output concat.
  - C2f/SPPF concats rewritten as split convs.
  - Result: COCO val2017 mAP50-95 44.31 → 43.86 (−0.45 pp), mAP50 60.70 → 60.39.
  - [Arm/yolov8s-int8-xnnpack-executorch](https://huggingface.co/Arm/yolov8s-int8-xnnpack-executorch)
- **Cloudflight YOLOv8n (COCO mAP50-95):**
  - FP32: 37.3
  - INT8 TFLite per-tensor with head: 32.9
  - INT8 TFLite per-channel: 33.9
  - INT8 TFLite with head kept FP32: 35.2
  - INT8 OpenVINO with head kept FP32: 37.1
  - They explain the failure mode: box coordinates (pixel scale) and class scores (0-1) are concatenated in one tensor, so a single INT8 scale destroys one of them. Excluding the final ops fixed it with little latency cost.
  - [Cloudflight engineering blog](https://engineering.cloudflight.io/quantized-yolo-for-edge-solutions)
- **Static INT8 (TensorRT) on YOLO models** loses roughly 3-7 points mAP50-95 on clean data. Degradation-aware calibration did not give consistent robustness gains under input degradations, which matters for murky underwater images. [arXiv 2508.19600](https://arxiv.org/abs/2508.19600v1)

### Inferences
- The user's "ORT static INT8 mixed precision" (2.1x) follows the published best practice. The Arm and Cloudflight data suggest that with head/DFL exclusion the accuracy cost should be under 1 pp mAP50-95 on COCO-like data. The project should still verify this on its gold test set, because small/low-contrast fish under turbid water may be more sensitive to quantization (cf. 2508.19600).
- The calibration set should come from the target domain (underwater frames, varied turbidity and lighting), not COCO.

### Gaps
- NCNN's INT8 tooling (`ncnn2table` / `ncnn2int8`, KL vs ACIQ vs EQ calibration) and its YOLOv8/11 accuracy were not retrieved from primary docs. Quality and head-exclusion support in ncnn's int8 tool are unverified.
- No QAT-vs-PTQ comparison for YOLO11/YOLO26 on ARM was found.
- No source specifically studies quantizing YOLO26's NMS-free one-to-one head.

## 4. Input resolution vs latency, and small-object recall; rectangular inference

### Takeaway
Latency on CPU scales roughly with pixel count. 320 is ~4x cheaper than 640, 480 ~1.8x, 512 ~1.6x, 416 ~2.4x. These are pixel-ratio estimates; Pi 5 detection measurements at multiple resolutions are scarce. Rectangular inputs (e.g. 640x480 or 640x384, multiples of 32) remove letterbox padding work.

### Cited Findings
- A Core Electronics guide reported that dropping from 640 to 320 "nearly quadrupled" FPS on Pi 5. This was for YOLO pose estimation, not detection, and comes from a search-result summary. [Core Electronics pose guide](https://core-electronics.com.au/guides/getting-started-with-yolo-pose-estimation-on-the-raspberry-pi/)
- Image sizes must be multiples of 32; a forum user only got a working 320 run after fixing this. [Core Electronics forum](https://forum.core-electronics.com.au/t/getting-started-with-yolo-object-and-animal-recognition-on-the-raspberry-pi/20923)
- Ultralytics NCNN export accepts `imgsz` as an int or an (h, w) tuple, so rectangular NCNN models (e.g. 480x640) can be exported directly. [Ultralytics NCNN integration](https://docs.ultralytics.com/integrations/ncnn/)

### Inferences
- **Pixel-ratio projection** (not measured), starting from the user's NCNN YOLO11n 70 ms at 640x640:
  - 640x480: ~53 ms
  - 512: ~45 ms
  - 480: ~40 ms
  - 416: ~30 ms
  - 320: ~18-20 ms
  - Fixed costs (pre/post-processing, thread sync) make the real gains smaller at low resolution.
- **Small-object recall.** Halving resolution halves the pixel size of each fish. Objects smaller than ~16-32 px at 640 will fall below the P3 (stride-8) detection floor at 320. Rectangular 640x480 (or 640x384 for 16:9) keeps full horizontal resolution and removes the 25-40% letterbox waste for 4:3/16:9 cameras. It is likely the best first step before lowering resolution.

### Gaps
- No published Pi 5 table of YOLO11/YOLO26 latency vs imgsz (320-640) for NCNN was found.
- No source quantifying small-object recall loss vs imgsz for YOLO11/26 was retrieved. The project's own gold set should be used to measure this.

## 5. Threading, governor, thermal throttling, power; sealed underwater housing

### Takeaway
Under sustained 4-core load an uncooled Pi 5 sits just above 85 °C and throttles continuously, dropping to about 1.5 GHz. With the fan disconnected, the heatsink-only Active Cooler starts throttling after ~200 s. Full-load board power is roughly 7-9 W (CPU stress), and more with peripherals. A sealed housing therefore needs a conductive path from the SoC to the hull or water. Otherwise sustained inference will throttle.

### Cited Findings
- **Throttling thresholds.** The firmware progressively throttles the Arm cores between 80 °C and 85 °C. At 85 °C both the Arm cores and the GPU are throttled. Users can lower the 85 °C limit but not raise it. Mode 3 is effectively always on for Pi 5 (DVFS setting removed). [Raspberry Pi documentation](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html)
- **Default fan curve (Active Cooler / case fan):** off below 50 °C, 30% at 50 °C, 50% at 60 °C, 70% at 67.5 °C, 100% at 75 °C, with 5 °C hysteresis. It is configurable via `fan_tempN`. The case fan draws ~60 mA at 5 V (0.3 W), from the USB current budget. [Raspberry Pi documentation](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html)
- **Official stress test** (`stress --cpu 4`, open bench, Oct 2023). [Raspberry Pi blog: Heating and cooling Raspberry Pi 5](https://www.raspberrypi.com/news/heating-and-cooling-raspberry-pi-5/)

  | Cooling | Idle temperature | Temperature under load | Throttling |
  |---|---|---|---|
  | None | ~65 °C | Just above 85 °C | Sustained |
  | Active Cooler | ~45 °C | ~60 °C (max 62-63 °C) | None |
  | Fan case | ~48 °C | 72-74 °C | None |
  | Active Cooler, fan unplugged | — | — | Starts ~T₀+200 s |

  - The blog says passive cooling is probably insufficient for heavy loads lasting more than 200-300 s.
  - A staff comment says throttled clocks drop from 2.4 GHz to 1.5 GHz.
- **Power, community measurements** (no first-party source found):
  - Idle headless: ~3.0 W.
  - All-core stress: ~8.8 W.
  - CPU + USB SSD + 4K video: up to ~16 W.
  - Active cooler adds ~0.5 W; NVMe adds 1-2 W.
  - The same site elsewhere lists 12-25 W "under load" for the 8 GB model, which conflicts with the 8.8 W figure; treat these as ranges.
  - [raspberry.tips power comparison](https://raspberry.tips/?p=9083); conflicting [raspberry.tips table](https://raspberry.tips/en/?p=8794)
  - ~12 W sustained at full CPU with NVMe and cooler. [fast.io guide](https://www.fast.io/resources/openclaw-raspberry-pi-power-supply-always-on-agent-guide.md)
- With the default `POWER_OFF_ON_HALT=0`, a Pi 5 draws 1.2-1.6 W even when shut down. [Jeff Geerling](https://www.jeffgeerling.com/blog/2023/reducing-raspberry-pi-5s-power-consumption-140x/)
- **Overclocking.** Ultralytics suggests 2.4 → 3.0 GHz CPU (step down in 100 MHz if unstable) and warns this increases heat and the risk of throttling. [Ultralytics Raspberry Pi guide](https://docs.ultralytics.com/guides/raspberry-pi/)
- **Threads.** ncnn's Pi 5 benchmark shows 4 threads vs 1 thread: yolov4-tiny 46.3 vs 68.0 ms (1.47x), resnet50 46.2 vs 89.4 ms (1.94x), nanodet_m 11.2 vs 19.7 ms (1.76x). Scaling from 1 to 4 threads is well below linear for small detectors. [ncnn benchmark README](https://github.com/Tencent/ncnn/blob/master/benchmark/README.md)

### Inferences
- **Inside a sealed housing**, continuous YOLO inference (an all-core load like `stress`) will reach the 80-85 °C throttle region, because internal air quickly reaches equilibrium. An Active Cooler alone just recirculates hot air. Budget ~6-9 W of heat to reject, and plan a thermal pad or heat spreader from the SoC to an aluminium end-cap or hull in contact with water. Water is an excellent heat sink if the conduction path exists.
- **If throttling is unavoidable**, run at 3 threads, at a lower resolution, or with frame skipping to keep the duty cycle below 100%. Expect at most a ~1.6x slowdown (2.4 → 1.5 GHz) in the worst sustained-throttle case.
- **CPU governor.** Use the `performance` governor (or keep the default ondemand pinned high) so per-frame latency is not hurt by frequency ramp-up. No Pi 5-specific measurement was found.

### Gaps
- No first-party Raspberry Pi measurement of Pi 5 power under sustained 4-core load was found (the official blog reported none). Community figures conflict: 8.8 W vs 12 W+.
- No source measured Pi 5 temperatures in a sealed enclosure, or during sustained YOLO inference specifically, as opposed to `stress`.
- No measured data on the governor's effect on inference latency for Pi 5.

## 6. Pipelining: parallel capture/preprocess/inference/postprocess, frame skipping, batching, using all 4 cores

### Takeaway
No Pi 5-specific published benchmarks of pipelined YOLO were found. The evidence above (sub-linear 1→4 thread scaling in ncnn) implies that throughput (FPS) can exceed 1/latency by running inference on 3 threads and doing capture/pre/post-processing on the 4th core, or by running two inference instances at 2 threads each.

### Cited Findings
- **Sub-linear thread scaling in ncnn on Pi 5.** 4 threads only give 1.47-1.94x over 1 thread on detection/backbone models. This leaves throughput headroom for pipelining or multi-instance layouts. [ncnn benchmark README](https://github.com/Tencent/ncnn/blob/master/benchmark/README.md)
- Ultralytics NCNN export supports `batch` (max images processed concurrently in predict). Ultralytics publishes no Pi 5 batch>1 numbers. [Ultralytics NCNN integration](https://docs.ultralytics.com/integrations/ncnn/)
- Ultralytics' Pi tables exclude pre/post-processing time. This explains why end-to-end numbers (like the user's 70 ms) differ from table numbers, and why moving these stages off the inference thread matters. [Ultralytics Raspberry Pi guide](https://docs.ultralytics.com/guides/raspberry-pi/)

### Inferences
- **Recommended pipeline:**
  - Thread A: capture via Picamera2/libcamera, zero-copy where possible.
  - Thread B: resize/letterbox to the rectangular model input, done in NumPy/OpenCV with NEON.
  - Thread C: NCNN inference on 3-4 threads.
  - Thread D: decode, NMS (none for YOLO26) and logging.
  - Connect them with bounded queues of size 1-2, dropping stale frames so latency stays low.
- **Batching** (batch>1) on CPU rarely improves per-frame throughput much once all cores are already busy, and it adds latency. It is likely not worth it for real-time use.
- **Frame skipping or a motion gate** (run the detector only when frame differences exceed a threshold, track in between) is the cheapest way to cut average CPU load and heat in a sealed housing.
- **What an accelerator would change** (out of scope, briefly): an NPU HAT would move inference off the CPU entirely. The CPU would then handle only capture and post-processing, reducing thermal load. Pi 5 numbers with such a HAT were not researched here.

### Gaps
- No published Pi 5 measurement of single-instance vs multi-instance NCNN YOLO throughput, or of pipelined vs sequential FPS.
- No published Pi 5 measurement of batch size effect for YOLO on CPU.
