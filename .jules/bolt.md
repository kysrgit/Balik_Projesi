# Bolt's Journal - Balik_Projesi Performance Learnings

## 2025-12-13 - Initial Codebase Analysis
**Context:** Real-time YOLO-based pufferfish detection system for Raspberry Pi 5 and PC.

### Key Performance Bottlenecks Identified:
1. **`frame.copy()` called unnecessarily** - In live_monitor.py, display_frame copies frame on EVERY frame even when no detection. Only needs copy when drawing boxes.
2. **Box coordinate conversion inefficient** - `.cpu().numpy()` called inside loops for each box, causing GPU-CPU transfer overhead.
3. **Missing frame skip optimization** - On low-end hardware (Pi), processing every frame is wasteful. Should consider adaptive frame skipping.

---

## 2026-07-29 - Vectorized YOLO Box Extraction & Zero-Copy Frame Handling
**Learning:** Iterating over Ultralytics `results[0].boxes` to call `.cpu().numpy()` on individual bounding box elements inside a Python loop creates 2*N C++/Python boundary transitions and Python scalar operations. Bulk converting `boxes.xyxy` and `boxes.conf` to NumPy arrays in one call and applying NumPy vectorized matrix multiplication eliminates loop overhead. Additionally, returning `frame` directly when `boxes` is empty in `draw_boxes()` avoids ~1-3 MB/frame memory copy allocations on empty detection frames.
**Action:** Always batch GPU/Tensor CPU transfers outside loops using `boxes.xyxy.cpu().numpy()` and avoid allocating frame copies on empty detection passes.
