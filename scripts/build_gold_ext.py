"""Dis negatif kontrol kumesi gold_v1_ext (Faz 2, ADR-002).

Amac: Egitimdeki zor negatifler (iNat Siganus/Sparisoma) ile gold_v1'deki negatifler
ayni kaynaktan geldigi icin, alarm dususunun "tur ezberi" olup olmadigini sinamak.
Kaynak: IMEDEA Labelled Mediterranean Fish Dataset (Zenodo 7534425, CC BY 4.0,
Mallorca, 25 sinif, hicbiri puffer degil). Tum goruntuler NEGATIF kabul edilir.

Yontem: pHash ile neredeyse ayni kareler ayiklanir (sabit kamera dizileri), sonra
sabit tohumla --n goruntu secilir. gold_v1 gibi dondurulur (SHA-256 listesi).

Ciktilar:
  dataset_gold_v1_ext/{images,labels}/, data.yaml   (labels bos)
  dataset_meta/gold_v1_ext.csv

Kullanim (GPU ortami: imagehash):
    .venv_gpu/Scripts/python.exe scripts/build_gold_ext.py
"""
from __future__ import annotations
import argparse, csv, hashlib, random, shutil
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "dataset_ext" / "imedea" / "raw" / "DEEPECOMAR_DATASET_YOLO"
OUT = ROOT / "dataset_gold_v1_ext"
META = ROOT / "dataset_meta" / "gold_v1_ext.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--dup-dist", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if META.exists() and not args.force:
        raise SystemExit("gold_v1_ext zaten dondurulmus.")

    imgs = sorted(p for p in SRC.rglob("*.jpg"))
    hashes = []
    for p in imgs:
        with Image.open(p) as im:
            hashes.append(int(str(imagehash.phash(im.convert("RGB"))), 16))
    h = np.array(hashes, dtype=np.uint64)
    order = list(range(len(imgs)))
    random.Random(2026).shuffle(order)
    kept = []
    for i in order:  # aç gozlu: secilmislerden hicbirine yakin olmayani al
        if kept:
            x = np.bitwise_xor(h[kept], h[i])
            dist = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
            if dist.min() <= args.dup_dist:
                continue
        kept.append(i)
        if len(kept) >= args.n:
            break
    print(f"{len(imgs)} goruntu, kopya ayiklandiktan sonra secilen: {len(kept)}")

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "images").mkdir(parents=True); (OUT / "labels").mkdir(parents=True)
    rows = []
    for k, i in enumerate(sorted(kept)):
        src = imgs[i]
        name = f"x{k:04d}.jpg"
        cv2.imwrite(str(OUT / "images" / name), cv2.imread(str(src)), [cv2.IMWRITE_JPEG_QUALITY, 95])
        (OUT / "labels" / f"x{k:04d}.txt").write_text("", encoding="utf-8")
        lbl = src.parent.parent / "labels" / f"{src.stem}.txt"
        n_fish = len([l for l in lbl.read_text(encoding="utf-8").splitlines() if l.strip()]) if lbl.exists() else 0
        rows.append({"file": name, "sha256": hashlib.sha256((OUT / "images" / name).read_bytes()).hexdigest(),
                     "source": src.relative_to(ROOT).as_posix(), "fish_in_source_labels": n_fish,
                     "license": "CC BY 4.0, IMEDEA Labelled Mediterranean Fish Dataset (Zenodo 7534425)"})
    (OUT / "data.yaml").write_text("train: images\nval: images\ntest: images\nnames:\n  0: pufferfish\n", encoding="utf-8")
    with META.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"gold_v1_ext: {len(rows)} negatif; kaynakta balik iceren: {sum(1 for r in rows if r['fish_in_source_labels'])}")


if __name__ == "__main__":
    main()
