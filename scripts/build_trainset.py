"""Faz 2 egitim veri varyantlarini olusturur (dataset_v2).

Kurallar:
  - Altin set (gold_v1) ile ayni pHash grubunda ya da ayni iNaturalist gozleminde olan
    hicbir goruntu egitime/dogrulamaya girmez.
  - Yeni dogrulama seti (val_v2) mevcut train'den KAYNAK duzeyinde ayrilir (%15; ayni
    fotografin Roboflow kopyalari ayni tarafta kalir) + zor negatiflerin %15'i.
    Model secimi yalnizca val_v2 ile yapilir; gold_v1'e en sonda bir kez bakilir.
  - Varyantlar yalnizca liste dosyalaridir (goruntuler kopyalanmaz); ultralytics
    etiketi /images/ -> /labels/ degisimiyle bulur.

Varyantlar (tek degisken):
  E2-0 base     : mevcut train (sentetik + kirpinti negatifler dahil)
  E2-1 hardneg  : base + iNaturalist zor negatifleri (Siganus, Sparisoma)
  E2-2 blur     : base + egitim goruntulerinin %50'sinin hareket-bulanik kopyasi
  E2-3 full     : base - sentetik negatifler + zor negatif + bulanik kopyalar

Kullanim:
    python scripts/build_trainset.py
"""
from __future__ import annotations
import csv, json, random, re, shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "dataset_meta"
OUT = ROOT / "dataset_v2"
SEED = 7
VAL_FRAC = 0.15


def read_csv(p):
    with p.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def motion_blur(img, rng):
    # Altin setin bulanik kopyasindan (k=9-17) daha genis bir dagilim: k=5-21
    k = int(rng.integers(5, 22)) | 1
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    m = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), float(rng.uniform(0, 180)), 1.0)
    kernel = cv2.warpAffine(kernel, m, (k, k))
    return cv2.filter2D(img, -1, kernel / max(kernel.sum(), 1e-6))


def main() -> None:
    rng = random.Random(SEED)
    nrng = np.random.default_rng(SEED)
    audit = read_csv(META / "audit.csv")
    gold = read_csv(META / "gold_v1.csv")
    by_path = {a["path"]: a for a in audit}
    gold_groups = {by_path[g["source"]]["dup_group"] for g in gold if g["source"] in by_path}
    photos = {p["file"]: p for p in read_csv(META / "inat_photos.csv")}
    gold_obs = {photos[g["source"]]["observation_id"] for g in gold if g["source"] in photos}

    # --- mevcut train: kaynak duzeyinde val ayrimi ---
    train = [a for a in audit if a["split"] == "train" and a["dup_group"] not in gold_groups]
    sources = sorted({a["source"] for a in train})
    rng.shuffle(sources)
    val_src = set(sources[: int(len(sources) * VAL_FRAC)])
    tr, va = [], []
    for a in train:
        (va if a["source"] in val_src else tr).append(a)
    # val'da ayni fotografin Roboflow kopyalarindan yalnizca biri kalsin
    seen, va1 = set(), []
    for a in sorted(va, key=lambda a: a["file"]):
        if a["source"] not in seen:
            seen.add(a["source"]); va1.append(a)

    # --- zor negatifler (gold'da kullanilmayan gozlemler) ---
    neg_dir = OUT / "extra" / "hardneg"
    for d in ("images", "labels"):
        (neg_dir / d).mkdir(parents=True, exist_ok=True)
    negs = [p for p in photos.values() if p["role"] == "negatif" and p["observation_id"] not in gold_obs
            and (ROOT / p["file"]).exists()]
    negs.sort(key=lambda p: p["file"])
    rng.shuffle(negs)
    n_val_neg = int(len(negs) * VAL_FRAC)
    neg_paths = []
    for p in negs:
        dst = neg_dir / "images" / Path(p["file"]).name
        if not dst.exists():
            im = cv2.imread(str(ROOT / p["file"]))
            h, w = im.shape[:2]
            s = 1280 / max(h, w)
            if s < 1:
                im = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(dst), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (neg_dir / "labels" / f"{dst.stem}.txt").write_text("", encoding="utf-8")
        neg_paths.append(dst)
    neg_val, neg_tr = neg_paths[:n_val_neg], neg_paths[n_val_neg:]

    # --- bulanik kopyalar (yalnizca egitim tarafindan, %50) ---
    blur_dir = OUT / "extra" / "blur"
    for d in ("images", "labels"):
        (blur_dir / d).mkdir(parents=True, exist_ok=True)
    blur_src = [a for a in tr if rng.random() < 0.5]
    blur_paths = []
    for a in blur_src:
        src = ROOT / a["path"]
        dst = blur_dir / "images" / f"blur_{src.name}"
        if not dst.exists():
            cv2.imwrite(str(dst), motion_blur(cv2.imread(str(src)), nrng), [cv2.IMWRITE_JPEG_QUALITY, 92])
        lbl = src.parent.parent / "labels" / f"{src.stem}.txt"
        shutil.copy(lbl, blur_dir / "labels" / f"{dst.stem}.txt") if lbl.exists() else \
            (blur_dir / "labels" / f"{dst.stem}.txt").write_text("", encoding="utf-8")
        blur_paths.append(dst)

    is_synth = lambda a: a["file"].startswith("neg_synth")
    base = [ROOT / a["path"] for a in tr]
    variants = {
        "E2-0_base": base,
        "E2-1_hardneg": base + neg_tr,
        "E2-2_blur": base + blur_paths,
        "E2-3_full": [ROOT / a["path"] for a in tr if not is_synth(a)] + neg_tr
                     + [b for b, a in zip(blur_paths, blur_src) if not is_synth(a)],
    }
    val = [ROOT / a["path"] for a in va1 if not is_synth(a)] + neg_val
    (OUT / "val_v2.txt").write_text("\n".join(p.as_posix() for p in val) + "\n", encoding="utf-8")
    summary = {"val_v2": {"images": len(val), "neg": len(neg_val)}}
    for name, paths in variants.items():
        (OUT / f"{name}.txt").write_text("\n".join(p.as_posix() for p in paths) + "\n", encoding="utf-8")
        (OUT / f"{name}.yaml").write_text(
            f"path: {OUT.as_posix()}\ntrain: {name}.txt\nval: val_v2.txt\nnames:\n  0: pufferfish\n", encoding="utf-8")
        summary[name] = {"images": len(paths)}
    summary["notes"] = {"train_sources": len(sources) - len(val_src), "val_sources": len(val_src),
                        "hardneg_train": len(neg_tr), "blur_copies": len(blur_paths),
                        "synthetic_removed_in_full": sum(1 for a in tr if is_synth(a))}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
