#!/usr/bin/env python3
"""
Hard Negative Mining Script
============================
Balon balığı modelinin false positive oranını düşürmek için
3 kaynaktan negatif örnek toplar:

1. WEBCAM HARD NEGATIVES — İnsan yüzü, el, oda, masa (doğrudan FP kaynağı)
2. BACKGROUND CROPS — Mevcut eğitim görsellerinden balık-dışı bölge kırpımları
3. SENTETIK NEGATIVES — Sualtı renk/doku simülasyonu, yuvarlak nesneler, gürültü

Hedef: %8-10 negatif oran → ~170 train + ~17 valid
Her negatif görsel için BOŞ .txt label dosyası oluşturulur (YOLO standardı).

Kullanım:
    python training/collect_negatives.py --webcam 50 --crop 80 --synthetic 60
"""
import os
import sys
import cv2
import glob
import time
import random
import argparse
import numpy as np
from pathlib import Path

# Proje kök dizini
ROOT = Path(__file__).parent.parent
DATASET = ROOT / "dataset"
TRAIN_IMG = DATASET / "train" / "images"
TRAIN_LBL = DATASET / "train" / "labels"
VALID_IMG = DATASET / "valid" / "images"
VALID_LBL = DATASET / "valid" / "labels"

# Negatif isim ön eki (çakışma önleme)
PREFIX = "neg_"


def count_existing():
    """Mevcut dataset istatistiklerini göster"""
    train_imgs = len(list(TRAIN_IMG.glob("*.*")))
    train_neg = sum(1 for f in TRAIN_LBL.glob("*.txt") if f.stat().st_size == 0)
    valid_imgs = len(list(VALID_IMG.glob("*.*")))
    valid_neg = sum(1 for f in VALID_LBL.glob("*.txt") if f.stat().st_size == 0)
    
    print("=" * 60)
    print("MEVCUT DATASET DURUMU")
    print("=" * 60)
    print(f"  Train: {train_imgs} görsel, {train_neg} negatif ({train_neg/max(train_imgs,1)*100:.1f}%)")
    print(f"  Valid: {valid_imgs} görsel, {valid_neg} negatif ({valid_neg/max(valid_imgs,1)*100:.1f}%)")
    print(f"  Hedef: en az %8 negatif")
    print("=" * 60)
    return train_imgs, train_neg, valid_imgs, valid_neg


def _save_negative(img, dest_img_dir, dest_lbl_dir, name):
    """Negatif görseli ve boş label dosyasını kaydet"""
    img_path = dest_img_dir / f"{name}.jpg"
    lbl_path = dest_lbl_dir / f"{name}.txt"
    
    # 640x640'a resize (eğitim boyutu)
    resized = cv2.resize(img, (640, 640))
    cv2.imwrite(str(img_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    # Boş label = negatif örnek
    lbl_path.touch()
    return True


# ─────────────────────────────────────────────────────────────────
# 1. WEBCAM HARD NEGATIVES
# ─────────────────────────────────────────────────────────────────
def collect_webcam(count=50, delay=0.3):
    """
    Laptop kamerasından hard negative frame'leri yakala.
    
    Stratejiler (her biri farklı FP senaryosu):
    - Düz yüz (en sık FP)
    - Yakın yüz (zoom)
    - El/parmaklar
    - Oda/masa/duvar
    - Kısmi yüz (profil)
    """
    print(f"\n📸 WEBCAM NEGATIVES: {count} frame yakalanacak")
    print("  Kamerayla çeşitli açılardan görüntü yakalanıyor...")
    print("  (Yüzünüzü, ellerinizi, masanızı, odanızı gösterin)")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  ❌ Webcam açılamadı! Atlaniyor...")
        return 0
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # İlk birkaç frame'i atla (kamera ısınması)
    for _ in range(10):
        cap.read()
    
    saved = 0
    aug_funcs = [
        lambda f: f,                                    # Ham frame
        lambda f: cv2.flip(f, 1),                       # Yatay flip
        lambda f: cv2.GaussianBlur(f, (5, 5), 0),      # Bulanık
        lambda f: _adjust_brightness(f, 0.6),           # Karanlık
        lambda f: _adjust_brightness(f, 1.4),           # Parlak
        lambda f: _add_underwater_tint(f),              # Sualtı renk kayması
        lambda f: _random_crop_resize(f),               # Rastgele kırpım
    ]
    
    for i in range(count):
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        
        # Her frame'e rastgele bir augmentasyon uygula
        aug = random.choice(aug_funcs)
        frame = aug(frame)
        
        # %85 train, %15 valid
        if random.random() < 0.85:
            dest_img, dest_lbl = TRAIN_IMG, TRAIN_LBL
        else:
            dest_img, dest_lbl = VALID_IMG, VALID_LBL
        
        name = f"{PREFIX}webcam_{i:04d}"
        if _save_negative(frame, dest_img, dest_lbl, name):
            saved += 1
        
        time.sleep(delay)  # Farklı kareler yakalamak için bekle
    
    cap.release()
    print(f"  ✅ {saved} webcam negatif kaydedildi")
    return saved


# ─────────────────────────────────────────────────────────────────
# 2. BACKGROUND CROPS
# ─────────────────────────────────────────────────────────────────
def collect_background_crops(count=80):
    """
    Mevcut eğitim görsellerinden balık İÇERMEYEN bölgeleri kırp.
    Polygon label'larından balığın olduğu bölgeyi tespit edip
    geri kalan arka plandan crop al.
    """
    print(f"\n🖼️  BACKGROUND CROPS: {count} kırpım oluşturulacak")
    
    # Etiketli görselleri bul
    train_images = list(TRAIN_IMG.glob("*.jpg")) + list(TRAIN_IMG.glob("*.png"))
    train_images = [p for p in train_images if not p.stem.startswith(PREFIX)]
    
    if len(train_images) == 0:
        print("  ❌ Eğitim görseli bulunamadı!")
        return 0
    
    random.shuffle(train_images)
    saved = 0
    
    for img_path in train_images:
        if saved >= count:
            break
        
        # Label dosyasını oku
        lbl_path = TRAIN_LBL / f"{img_path.stem}.txt"
        if not lbl_path.exists() or lbl_path.stat().st_size == 0:
            continue
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        # Polygon'dan bbox hesapla (balığın olduğu bölge)
        fish_boxes = _parse_polygon_bbox(lbl_path, w, h)
        if not fish_boxes:
            continue
        
        # Balığın OLMADIĞI rastgele bölgeler kırp
        for attempt in range(5):
            crop = _random_non_fish_crop(img, fish_boxes, min_size=160)
            if crop is not None:
                if random.random() < 0.85:
                    dest_img, dest_lbl = TRAIN_IMG, TRAIN_LBL
                else:
                    dest_img, dest_lbl = VALID_IMG, VALID_LBL
                
                name = f"{PREFIX}bgcrop_{saved:04d}"
                
                # Augmentasyon
                aug = random.choice([
                    lambda f: f,
                    lambda f: _add_underwater_tint(f),
                    lambda f: cv2.GaussianBlur(f, (3, 3), 0),
                    lambda f: _adjust_brightness(f, random.uniform(0.5, 1.3)),
                ])
                crop = aug(crop)
                
                if _save_negative(crop, dest_img, dest_lbl, name):
                    saved += 1
                break
    
    print(f"  ✅ {saved} background crop kaydedildi")
    return saved


def _parse_polygon_bbox(lbl_path, img_w, img_h):
    """YOLO polygon label'dan bounding box listesi çıkar"""
    boxes = []
    with open(lbl_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            
            # İlk eleman class_id, sonrası x,y çiftleri
            coords = list(map(float, parts[1:]))
            xs = [coords[i] * img_w for i in range(0, len(coords), 2)]
            ys = [coords[i] * img_h for i in range(1, len(coords), 2)]
            
            if xs and ys:
                x1 = max(0, int(min(xs)) - 20)
                y1 = max(0, int(min(ys)) - 20)
                x2 = min(img_w, int(max(xs)) + 20)
                y2 = min(img_h, int(max(ys)) + 20)
                boxes.append((x1, y1, x2, y2))
    
    return boxes


def _random_non_fish_crop(img, fish_boxes, min_size=160):
    """Balık kutularıyla çakışmayan rastgele dikdörtgen kırp"""
    h, w = img.shape[:2]
    
    for _ in range(20):  # Max 20 deneme
        crop_size = random.randint(min_size, min(h, w, 400))
        cx = random.randint(0, w - crop_size)
        cy = random.randint(0, h - crop_size)
        
        # IoU kontrolü — hiçbir fish box ile çakışmamalı
        overlaps = False
        for (fx1, fy1, fx2, fy2) in fish_boxes:
            # Çakışma kontrolü
            if cx < fx2 and cx + crop_size > fx1 and cy < fy2 and cy + crop_size > fy1:
                overlaps = True
                break
        
        if not overlaps:
            return img[cy:cy+crop_size, cx:cx+crop_size]
    
    return None


# ─────────────────────────────────────────────────────────────────
# 3. SENTETIK NEGATIVES
# ─────────────────────────────────────────────────────────────────
def collect_synthetic(count=60):
    """
    OpenCV ile sentetik negatif görseller üret.
    
    Kategoriler:
    - Sualtı arka planları (mavi/yeşil gradyanlar)
    - Yuvarlak/oval nesneler (FP tetikleyicisi)
    - Rastgele doku ve gürültü
    - İnsan cildi tonlarında yüzeyler
    - Deniz tabanı simülasyonu (kum, kaya)
    """
    print(f"\n🎨 SENTETIK NEGATIVES: {count} görsel üretilecek")
    
    generators = [
        _gen_underwater_background,
        _gen_round_objects,
        _gen_noise_texture,
        _gen_skin_tone_surface,
        _gen_seabed,
        _gen_water_surface,
        _gen_coral_rocks,
        _gen_gradient_blob,
    ]
    
    saved = 0
    for i in range(count):
        gen = random.choice(generators)
        img = gen(640, 640)
        
        if random.random() < 0.85:
            dest_img, dest_lbl = TRAIN_IMG, TRAIN_LBL
        else:
            dest_img, dest_lbl = VALID_IMG, VALID_LBL
        
        name = f"{PREFIX}synth_{i:04d}"
        if _save_negative(img, dest_img, dest_lbl, name):
            saved += 1
    
    print(f"  ✅ {saved} sentetik negatif kaydedildi")
    return saved


def _gen_underwater_background(w, h):
    """Sualtı mavi/yeşil gradyan arka plan"""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Üstten mavi → alttan yeşil/koyu gradyan
    for y in range(h):
        ratio = y / h
        b = int(180 - ratio * 80 + random.randint(-15, 15))
        g = int(120 + ratio * 40 + random.randint(-15, 15))
        r = int(40 + ratio * 20 + random.randint(-10, 10))
        img[y, :] = [max(0, min(255, b)), max(0, min(255, g)), max(0, min(255, r))]
    
    # Rastgele partiküller (sualtı parçacıkları)
    for _ in range(random.randint(20, 80)):
        x = random.randint(0, w-1)
        y = random.randint(0, h-1)
        size = random.randint(1, 4)
        color = (200 + random.randint(-30, 30), 200 + random.randint(-30, 30), 180 + random.randint(-20, 20))
        color = tuple(max(0, min(255, c)) for c in color)
        cv2.circle(img, (x, y), size, color, -1)
    
    # Hafif bulanıklık
    img = cv2.GaussianBlur(img, (5, 5), 1.5)
    return img


def _gen_round_objects(w, h):
    """Yuvarlak/oval nesneler — balon balığı şeklini taklit eden ama olmayan"""
    bg = _gen_underwater_background(w, h)
    
    n_objects = random.randint(1, 4)
    for _ in range(n_objects):
        cx = random.randint(100, w-100)
        cy = random.randint(100, h-100)
        rx = random.randint(30, 120)
        ry = random.randint(20, 80)
        angle = random.randint(0, 180)
        
        # Farklı renkler (kaya, mercan, yosun tonları — balık DEĞİL)
        colors = [
            (80, 80, 80),    # Gri kaya
            (50, 100, 80),   # Yeşil yosun
            (60, 60, 120),   # Kahverengi kaya
            (100, 120, 140), # Kumlu
            (70, 90, 60),    # Koyu yeşil
        ]
        color = random.choice(colors)
        color = tuple(c + random.randint(-20, 20) for c in color)
        color = tuple(max(0, min(255, c)) for c in color)
        
        cv2.ellipse(bg, (cx, cy), (rx, ry), angle, 0, 360, color, -1)
        
        # Hafif kenar yumuşatma
        bg = cv2.GaussianBlur(bg, (3, 3), 0.8)
    
    return bg


def _gen_noise_texture(w, h):
    """Rastgele gürültü dokusu — model robustluğu için"""
    noise = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    
    # Sualtı renk kayması ekle
    noise = _add_underwater_tint(noise)
    
    # Bulanıklaştır
    ksize = random.choice([3, 5, 7])
    noise = cv2.GaussianBlur(noise, (ksize, ksize), 0)
    
    return noise


def _gen_skin_tone_surface(w, h):
    """İnsan cildi tonlarında yüzey — yüz FP'lerini azaltmak için"""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Çeşitli cilt tonları
    skin_tones = [
        (160, 180, 210),  # Açık ten
        (120, 140, 170),  # Orta ten
        (80, 100, 140),   # Koyu ten
        (140, 160, 190),  # Pembe ten
        (100, 120, 150),  # Esmer
    ]
    
    base = random.choice(skin_tones)
    img[:] = base
    
    # Gradyan ve varyasyon ekle
    for y in range(h):
        for x in range(0, w, 10):
            noise = [random.randint(-15, 15) for _ in range(3)]
            end_x = min(x + 10, w)
            for ch in range(3):
                img[y, x:end_x, ch] = max(0, min(255, base[ch] + noise[ch]))
    
    # Dokusal varyasyon
    texture = np.random.randint(-10, 10, (h, w, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + texture, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (7, 7), 2)
    
    return img


def _gen_seabed(w, h):
    """Deniz tabanı simülasyonu — kum, çakıl, kaya"""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Kum taban rengi
    base_b = random.randint(100, 140)
    base_g = random.randint(130, 170)
    base_r = random.randint(140, 180)
    img[:] = (base_b, base_g, base_r)
    
    # Perlin-benzeri kum dokusu (basit yaklaşım)
    noise = np.random.randint(-25, 25, (h//4, w//4, 3), dtype=np.int16)
    noise = cv2.resize(noise.astype(np.float32), (w, h)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Rastgele taşlar
    for _ in range(random.randint(5, 20)):
        cx = random.randint(0, w)
        cy = random.randint(0, h)
        rx = random.randint(10, 50)
        ry = random.randint(8, 35)
        gray = random.randint(60, 120)
        cv2.ellipse(img, (cx, cy), (rx, ry), random.randint(0, 180), 0, 360, (gray, gray+10, gray+5), -1)
    
    img = cv2.GaussianBlur(img, (5, 5), 1.2)
    return img


def _gen_water_surface(w, h):
    """Su yüzeyi — ışık yansımaları ile"""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Mavi su
    img[:] = (180, 130, 50)
    
    # Dalgalı ışık desenleri
    for y in range(h):
        wave = int(10 * np.sin(y / 20 + random.random() * 10))
        brightness = int(20 * np.sin(y / 40 + random.random() * 5))
        for x in range(w):
            wave_x = int(8 * np.sin(x / 25 + y / 30))
            b = min(255, max(0, 180 + wave + wave_x + brightness + random.randint(-5, 5)))
            g = min(255, max(0, 130 + wave//2 + random.randint(-5, 5)))
            r = min(255, max(0, 50 + random.randint(-5, 5)))
            img[y, x] = (b, g, r)
    
    # Işık kabarcıkları
    for _ in range(random.randint(3, 12)):
        cx, cy = random.randint(0, w), random.randint(0, h)
        r = random.randint(5, 25)
        cv2.circle(img, (cx, cy), r, (220, 200, 150), -1)
    
    img = cv2.GaussianBlur(img, (5, 5), 1.5)
    return img


def _gen_coral_rocks(w, h):
    """Mercan ve kaya formasyonları"""
    bg = _gen_underwater_background(w, h)
    
    # Düzensiz mercan/kaya şekilleri
    for _ in range(random.randint(3, 10)):
        n_pts = random.randint(5, 12)
        cx = random.randint(50, w-50)
        cy = random.randint(50, h-50)
        pts = []
        for j in range(n_pts):
            angle = 2 * np.pi * j / n_pts
            r = random.randint(20, 80)
            px = int(cx + r * np.cos(angle) + random.randint(-15, 15))
            py = int(cy + r * np.sin(angle) + random.randint(-15, 15))
            pts.append([px, py])
        
        pts = np.array(pts, dtype=np.int32)
        color = (
            random.randint(40, 120),
            random.randint(50, 130),
            random.randint(30, 100),
        )
        cv2.fillPoly(bg, [pts], color)
    
    bg = cv2.GaussianBlur(bg, (3, 3), 1.0)
    return bg


def _gen_gradient_blob(w, h):
    """Amorf gradyan blob — her türlü şekil"""
    bg = _gen_underwater_background(w, h)
    
    # Gaussians blob
    for _ in range(random.randint(2, 6)):
        cx = random.randint(0, w)
        cy = random.randint(0, h)
        sigma = random.randint(30, 150)
        
        Y, X = np.mgrid[0:h, 0:w]
        gauss = np.exp(-((X-cx)**2 + (Y-cy)**2) / (2 * sigma**2))
        
        color = np.array([random.randint(40, 200), random.randint(40, 200), random.randint(40, 150)])
        blob = (gauss[:, :, np.newaxis] * color * random.uniform(0.3, 0.8)).astype(np.uint8)
        bg = cv2.add(bg, blob)
    
    return bg


# ─────────────────────────────────────────────────────────────────
# Yardımcı Augmentasyon Fonksiyonları
# ─────────────────────────────────────────────────────────────────
def _adjust_brightness(img, factor):
    """Parlaklık ayarı"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _add_underwater_tint(img):
    """Sualtı renk kayması (kırmızı azaltma, mavi/yeşil artırma)"""
    result = img.copy().astype(np.int16)
    result[:, :, 0] = np.clip(result[:, :, 0] + random.randint(15, 40), 0, 255)  # Blue+
    result[:, :, 1] = np.clip(result[:, :, 1] + random.randint(5, 20), 0, 255)   # Green+
    result[:, :, 2] = np.clip(result[:, :, 2] - random.randint(15, 40), 0, 255)  # Red-
    return result.astype(np.uint8)


def _random_crop_resize(img):
    """Rastgele kırpım + resize"""
    h, w = img.shape[:2]
    crop_size = random.randint(min(h, w) // 3, min(h, w))
    x = random.randint(0, w - crop_size)
    y = random.randint(0, h - crop_size)
    crop = img[y:y+crop_size, x:x+crop_size]
    return cv2.resize(crop, (w, h))


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Hard Negative Mining")
    parser.add_argument("--webcam", type=int, default=50, help="Webcam frame sayısı (default: 50)")
    parser.add_argument("--crop", type=int, default=80, help="Background crop sayısı (default: 80)")
    parser.add_argument("--synthetic", type=int, default=60, help="Sentetik negatif sayısı (default: 60)")
    parser.add_argument("--no-webcam", action="store_true", help="Webcam kullanma")
    args = parser.parse_args()
    
    print("🐡 HARD NEGATIVE MINING — Balon Balığı Tespit Sistemi")
    print("=" * 60)
    
    # Mevcut durum
    train_imgs, train_neg, valid_imgs, valid_neg = count_existing()
    
    total_before = train_neg + valid_neg
    total_planned = 0
    
    # 1. Webcam
    webcam_count = 0
    if not args.no_webcam:
        webcam_count = collect_webcam(args.webcam, delay=0.2)
        total_planned += webcam_count
    
    # 2. Background crops
    crop_count = collect_background_crops(args.crop)
    total_planned += crop_count
    
    # 3. Synthetic
    synth_count = collect_synthetic(args.synthetic)
    total_planned += synth_count
    
    # Sonuç raporu
    print("\n" + "=" * 60)
    print("SONUÇ RAPORU")
    print("=" * 60)
    
    final_train_imgs, final_train_neg, final_valid_imgs, final_valid_neg = count_existing()
    
    print(f"\n  ÖNCE:")
    print(f"    Train: {train_imgs} görsel, {train_neg} negatif ({train_neg/max(train_imgs,1)*100:.1f}%)")
    print(f"    Valid: {valid_imgs} görsel, {valid_neg} negatif ({valid_neg/max(valid_imgs,1)*100:.1f}%)")
    
    print(f"\n  SONRA:")
    print(f"    Train: {final_train_imgs} görsel, {final_train_neg} negatif ({final_train_neg/max(final_train_imgs,1)*100:.1f}%)")
    print(f"    Valid: {final_valid_imgs} görsel, {final_valid_neg} negatif ({final_valid_neg/max(final_valid_imgs,1)*100:.1f}%)")
    
    added = (final_train_neg + final_valid_neg) - total_before
    print(f"\n  Toplam eklenen negatif: {added}")
    print(f"    Webcam: {webcam_count}")
    print(f"    Background crop: {crop_count}")
    print(f"    Sentetik: {synth_count}")
    
    overall_neg_pct = (final_train_neg + final_valid_neg) / max(final_train_imgs + final_valid_imgs, 1) * 100
    if overall_neg_pct >= 8:
        print(f"\n  ✅ Negatif oranı yeterli: %{overall_neg_pct:.1f}")
    else:
        print(f"\n  ⚠️  Negatif oranı hâlâ düşük: %{overall_neg_pct:.1f} (hedef: %8+)")
        print(f"      Tekrar çalıştırarak artırabilirsiniz.")
    
    print("\n  Sonraki adım: python training/train_yolo.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
