# Bolt's Journal - Balik_Projesi Performance Learnings

## 2025-12-13 - Initial Codebase Analysis
**Context:** Real-time YOLO-based pufferfish detection system for Raspberry Pi 5 and PC.

### Key Performance Bottlenecks Identified:
1. **`frame.copy()` called unnecessarily** - In live_monitor.py, display_frame copies frame on EVERY frame even when no detection. Only needs copy when drawing boxes.
2. **Box coordinate conversion inefficient** - `.cpu().numpy()` called inside loops for each box, causing GPU-CPU transfer overhead.
3. **Missing frame skip optimization** - On low-end hardware (Pi), processing every frame is wasteful. Should consider adaptive frame skipping.

---
