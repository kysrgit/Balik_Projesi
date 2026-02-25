# Goruntu isleme yardimcilari
import cv2

_clahe_cache = None
_last_clip = None
_last_grid = None

def apply_clahe(img, clip=3.0, grid=(8, 8)):
    """Lab renk uzayinda CLAHE uygula - sualti goruntuler icin"""
    global _clahe_cache, _last_clip, _last_grid
    
    if img is None:
        return None
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # ⚡ Bolt Optimization: Sadece ayarlar degistiginde obje yarat (C++ memory protection)
    if _clahe_cache is None or _last_clip != clip or _last_grid != grid:
        _clahe_cache = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
        _last_clip = clip
        _last_grid = grid
        
    l = _clahe_cache.apply(l)
    
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def draw_boxes(frame, boxes, confs, color=(0, 0, 255)):
    """Tespit kutularini ciz"""
    result = frame.copy()
    for (x1, y1, x2, y2), conf in zip(boxes, confs):
        c = (0, 0, 255) if conf > 0.85 else (0, 255, 255)
        cv2.rectangle(result, (x1, y1), (x2, y2), c, 2)
        cv2.putText(result, f"{conf:.2f}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    return result
