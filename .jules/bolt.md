## 2026-02-24 - [OpenCV Object Instantiation Overhead]
**Learning:** Instantiating `cv2.createCLAHE` repeatedly inside a 30 FPS camera producer loop introduces unnecessary C++ memory allocation and garbage collection overhead. While Python abstractions make it seem lightweight, the underlying C++ calls are relatively expensive.
**Action:** Always cache algorithm/filter objects in continuous processing loops (like `apply_clahe`) using global or class-level state, invalidating the cache only when filter hyperparameters (like `clipLimit`) are modified by the UI.
