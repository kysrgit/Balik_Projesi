# Raspberry Pi 5 ONNX Benchmark Summary

This benchmark was executed on Raspberry Pi 5.

## Benchmark Scope

- Date/time (local): 2026-10-14T23:59:31+03:00
- Benchmark mode: image_folder
- Primary metric scope: preprocessing + inference + postprocessing/NMS
- Model path: `/home/keysar/Balik_Projesi/models/pufferfish_s_v2_int8.onnx`
- Model file size (bytes): 12526461
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
| Mean latency (ms/frame) | 171.07 |
| Median latency (ms/frame) | 171.07 |
| p90 latency (ms/frame) | 171.46 |
| p95 latency (ms/frame) | 171.66 |
| Min latency (ms/frame) | 170.22 |
| Max latency (ms/frame) | 174.10 |
| Std. dev. latency (ms/frame) | 0.40 |
| Mean FPS | 5.85 |
| Median FPS | 5.85 |
| Effective end-to-end FPS (live camera) | N/A |
| Mean preprocessing time (ms) | 2.01 |
| Mean inference time (ms) | 167.91 |
| Mean postprocessing/NMS time (ms) | 1.15 |
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
| CPU temp before (C) | `53.45` |
| CPU temp after (C) | `63.90` |
| Throttled before | `throttled=0x0` |
| Throttled after | `throttled=0x0` |

## Artifact Files

- JSON: `/home/keysar/Balik_Projesi/scripts/benchmark_result_20261014_235931.json`
- CSV: `/home/keysar/Balik_Projesi/scripts/benchmark_result_20261014_235931.csv`
- Summary Markdown: `/home/keysar/Balik_Projesi/scripts/benchmark_summary_20261014_235931.md`

## Notes

- `Mean FPS` and `Mean latency` in this summary refer only to the declared primary metric scope above.
- `Pipeline` metrics isolate preprocessing + ONNX inference + postprocessing/NMS.
- `Effective end-to-end FPS (live camera)` is a wall-clock throughput figure and should not be confused with dashboard streaming FPS.
