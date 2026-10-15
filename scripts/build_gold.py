"""Altin test setini (gold_v1) olusturur ve dondurur (Faz 1, ADR-001).

Kurallar:
  - Yalnizca insan tarafindan dogrulanmis "suda" goruntuler (su_alti, su_ustu).
  - Mevcut test+valid: dataset_meta/env_tags.csv ile dogrulanmis, etiketi hatali isaretlenmemis.
  - iNaturalist: dataset_meta/inat_labels.json ile etiketlenmis (ok -> pozitif, puffer_yok -> negatif).
  - Egitim (train) setindeki herhangi bir goruntuyle gorsel kopya (pHash grubu) olan her sey disarida.
  - Ayni iNaturalist gozleminin fotograflari birlikte kalir (kaynak duzeyi).
  - Hareket bulanikligi kopyasi (ADR-000 K7): her goruntunun sabit tohumlu bulanik kopyasi ayri klasorde.

Ciktilar:
  dataset_gold_v1/{images,labels}/          ana altin set (YOLO kutu formati)
  dataset_gold_v1_blur/{images,labels}/     bulanik kopya
  dataset_meta/gold_v1.csv                  dosya listesi + sha256 + koken (git'te; seti dondurur)
  dataset_gold_v1/data.yaml, dataset_gold_v1_blur/data.yaml

Kullanim:
    python scripts/build_gold.py            # eksik dogrulama varsa durur ve listeler
    python scripts/build_gold.py --force    # dondurulmus seti yeniden yazar (yeni surum icin kullanin!)
"""
from __future__ import annotations
import argparse, csv, hashlib, json, shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "dataset_meta"
VERSION = "gold_v1"
OUT = ROOT / f"dataset_{VERSION}"
OUT_BLUR = ROOT / f"dataset_{VERSION}_blur"
WATER = {"su_alti", "su_ustu"}
# Arayuzdeki hata bildirimleri (dataset_meta/label_fixes.json)
EXCLUDE_ISSUES = {"yanlis_tur", "kopya", "kalite"}          # goruntu altin sete hic girmez
BOX_ISSUES = {"eksik_kutu", "fazla_kutu", "kayik_kutu"}      # kutu duzeltilmediyse girmez


def read_csv(p: Path) -> list[dict]:
    with p.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def poly_to_boxes(label: Path) -> list[list[float]]:
    out = []
    if not label.exists():
        return out
    for line in label.read_text(encoding="utf-8").splitlines():
        v = list(map(float, line.split()[1:]))
        if len(v) == 4:
            out.append(v)
        elif len(v) > 4:
            xs, ys = v[0::2], v[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            out.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
    return out


def xyxy_to_yolo(b):
    x1, y1, x2, y2 = b
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]


def motion_blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    k = int(rng.integers(9, 17)) | 1
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    angle = float(rng.uniform(0, 180))
    m = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, m, (k, k))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(img, -1, kernel)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if (META / f"{VERSION}.csv").exists() and not args.force:
        raise SystemExit(f"{VERSION} zaten dondurulmus. Degistirmek yerine yeni surum (gold_v2) olusturun.")

    audit = read_csv(META / "audit.csv")
    train_groups = {r["dup_group"] for r in audit if r["split"] == "train"}
    group_of = {r["path"]: r["dup_group"] for r in audit}

    items, problems = [], []
    # 1) Mevcut test + valid
    tags = {(t["split"], t["source"]): t for t in read_csv(META / "env_tags.csv")} if (META / "env_tags.csv").exists() else {}
    fixes = json.loads((META / "label_fixes.json").read_text(encoding="utf-8")) if (META / "label_fixes.json").exists() else {}
    for r in audit:
        if r["split"] not in ("test", "valid"):
            continue
        t = tags.get((r["split"], r["source"]))
        if t is None:
            problems.append(f"dogrulanmamis: {r['split']}:{r['source']}")
            continue
        fx = fixes.get(r["path"])  # arayuzden yapilan duzeltme / hata bildirimi
        env = (fx or {}).get("env") or t["env"]
        if env not in WATER or r["dup_group"] in train_groups:
            continue
        if fx and set(fx["issues"]) & EXCLUDE_ISSUES:
            continue
        stem = Path(r["file"]).stem
        lbl = ROOT / "dataset" / r["split"] / "labels" / f"{stem}.txt"
        if fx and fx.get("boxes") is not None:
            boxes = [xyxy_to_yolo(b) for b in fx["boxes"]]  # insan tarafindan duzeltilmis kutular
        elif t["etiket_hatali"] == "1" or (fx and set(fx["issues"]) & BOX_ISSUES):
            continue  # kutu hatali bildirildi ama duzeltilmedi
        else:
            boxes = poly_to_boxes(lbl)
        items.append({"src": ROOT / r["path"], "boxes": boxes, "origin": f"dataset/{r['split']}",
                      "env": env, "group": f"{r['split']}:{r['source']}", "license": "Roboflow CC BY 4.0"})

    # 2) iNaturalist
    labels = json.loads((META / "inat_labels.json").read_text(encoding="utf-8")) if (META / "inat_labels.json").exists() else {}
    photos = {p["file"]: p for p in read_csv(META / "inat_photos.csv")} if (META / "inat_photos.csv").exists() else {}
    for f, p in photos.items():
        lab = labels.get(f)
        if lab is None:
            problems.append(f"etiketlenmemis: {f}")
            continue
        if lab["status"] == "haric" or lab["env"] not in WATER:
            continue
        if group_of.get(f) in train_groups:
            continue  # zaten egitimde var -> sizinti
        boxes = [xyxy_to_yolo(b) for b in lab["boxes"]] if lab["status"] == "ok" else []
        items.append({"src": ROOT / f, "boxes": boxes, "origin": f"inat/{p['role']}", "env": lab["env"],
                      "group": f"inat:{p['observation_id']}", "license": f"{p['license']} {p['attribution']}"})

    if problems:
        print(f"{len(problems)} goruntu henuz dogrulanmadi/etiketlenmedi. Ilk 10:")
        print("\n".join("  " + x for x in problems[:10]))
        raise SystemExit("Once /denetim ve /etiket sayfalarini tamamlayin.")

    for d in (OUT, OUT_BLUR):
        if d.exists():
            shutil.rmtree(d)
        (d / "images").mkdir(parents=True); (d / "labels").mkdir(parents=True)
    rng = np.random.default_rng(2026)
    rows = []
    for k, it in enumerate(sorted(items, key=lambda i: str(i["src"]))):
        name = f"g{k:04d}"
        img = cv2.imread(str(it["src"]))
        cv2.imwrite(str(OUT / "images" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(OUT_BLUR / "images" / f"{name}.jpg"), motion_blur(img, rng), [cv2.IMWRITE_JPEG_QUALITY, 95])
        txt = "".join(f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n" for b in it["boxes"])
        for d in (OUT, OUT_BLUR):
            (d / "labels" / f"{name}.txt").write_text(txt, encoding="utf-8")
        sha = hashlib.sha256((OUT / "images" / f"{name}.jpg").read_bytes()).hexdigest()
        rows.append({"file": f"{name}.jpg", "sha256": sha, "origin": it["origin"], "source": it["src"].relative_to(ROOT).as_posix(),
                     "group": it["group"], "env": it["env"], "instances": len(it["boxes"]), "license": it["license"]})
    with (META / f"{VERSION}.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    # 'path' yazilmaz: ultralytics yaml'in bulundugu klasoru kok kabul eder (makineden bagimsiz)
    for d in (OUT, OUT_BLUR):
        (d / "data.yaml").write_text("train: images\nval: images\ntest: images\nnames:\n  0: pufferfish\n",
                                     encoding="utf-8")
    pos = sum(1 for r in rows if r["instances"])
    print(f"{VERSION}: {len(rows)} goruntu ({pos} pozitif, {len(rows) - pos} negatif), "
          f"{sum(r['instances'] for r in rows)} ornek; kokenler: "
          + ", ".join(f"{o}={sum(1 for r in rows if r['origin'] == o)}" for o in sorted({r['origin'] for r in rows})))


if __name__ == "__main__":
    main()
