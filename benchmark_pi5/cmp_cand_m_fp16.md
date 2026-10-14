# Raspberry Pi 5 ONNX Benchmark Summary

This benchmark was executed on Raspberry Pi 5.

## Benchmark Scope

- Date/time (local): 2026-10-15T00:27:59+03:00
- Benchmark mode: image_folder
- Primary metric scope: preprocessing + inference + postprocessing/NMS
- Model path: `/home/keysar/Balik_Projesi/models/candidates/cand_m_fp16.onnx`
- Model file size (bytes): 40282745
- Input size: 640 x 640
- Confidence threshold: 0.6
- CLAHE: off
- Execution provider requested: cpu
- Execution provider used: CPUExecutionProvider
- Available execution providers: AzureExecutionProvider, CPUExecutionProvider
- Warmup frames: 20
- Benchmark frames: 200
- Preprocessing included: yes
- Postprocessing/NMS included: yes
- Camera capture included: no

## Key Results

| Metric | Value |
| --- | --- |
| Mean latency (ms/frame) | 1373.36 |
| Median latency (ms/frame) | 1372.94 |
| p90 latency (ms/frame) | 1390.20 |
| p95 latency (ms/frame) | 1393.58 |
| Min latency (ms/frame) | 1338.75 |
| Max latency (ms/frame) | 1415.61 |
| Std. dev. latency (ms/frame) | 12.48 |
| Mean FPS | 0.73 |
| Median FPS | 0.73 |
| Effective end-to-end FPS (live camera) | N/A |
| Mean preprocessing time (ms) | 8.12 |
| Mean inference time (ms) | 1363.95 |
| Mean postprocessing/NMS time (ms) | 1.29 |
| Dropped frames | 0 |

## Platform Metadata

| Field | Value |
| --- | --- |
| Hostname | `keysar` |
| Platform | `Linux-6.18.50+rpt-rpi-2712-aarch64-with-glibc2.41` |
| OS | `Debian GNU/Linux 13 (trixie)` |
| Raspberry Pi model | `Raspberry Pi 5 Model B Rev 1.1` |
| CPU model | `N/A` |
| RAM (MB) | `16218` |
| Python | `3.13.5 (main, Aug 10 2026, 12:06:59) [GCC 14.2.0]` |
| ONNX Runtime | `1.31.0` |
| OpenCV | `5.0.0` |
| NumPy | `2.2.4` |
| CPU temp before (C) | `58.40` |
| CPU temp after (C) | `72.70` |
| Throttled before | `throttled=0x0` |
| Throttled after | `throttled=0x0` |

## Artifact Files

- JSON: `/home/keysar/Balik_Projesi/scripts/benchmark_result_20261015_002759.json`
- CSV: `/home/keysar/Balik_Projesi/scripts/benchmark_result_20261015_002759.csv`
- Summary Markdown: `/home/keysar/Balik_Projesi/scripts/benchmark_summary_20261015_002759.md`

## Notes

- `Mean FPS` and `Mean latency` in this summary refer only to the declared primary metric scope above.
- `Pipeline` metrics isolate preprocessing + ONNX inference + postprocessing/NMS.
- `Effective end-to-end FPS (live camera)` is a wall-clock throughput figure and should not be confused with dashboard streaming FPS.
