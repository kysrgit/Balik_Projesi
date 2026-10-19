"""Altin set adaylari icin BAGIMSIZ kutu onerileri (Faz 1).

Neden: Kutu onerisini yalnizca kendi modelimiz verirse, modelin kacirdigi baliklar
etiketlenmez ve altin set modelin zaten iyi bildigi goruntulere kayar (iyimser yanli
skor). Bu yuzden acik-sozluk bir dedektor (OWLv2, bizim veriyle hic egitilmedi) ile
"balik" onerileri uretilir ve YOLO onerileriyle birlestirilir; hangi kutunun puffer
oldugunu insan/denetci secer.

Aday secimi (sabit tohum):
  - pozitif: mevcut veri setinde kopyasi olmayan, CLIP'e gore "suda" goruntuler;
    tum Lagocephalus + rastgele --n-torq Torquigener; gozlem basina 1 fotograf
  - negatif: ayni kurallarla rastgele --n-neg goruntu

Cikti: dataset_meta/gold_candidates.json
    {dosya: {"role", "taxon", "boxes": [[x1,y1,x2,y2,kaynak,skor], ...]}}  (0-1 normalize)

Kullanim (GPU ortami):
    .venv_gpu/Scripts/python.exe scripts/propose_boxes.py
"""
from __future__ import annotations
import argparse, csv, json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "dataset_meta"
WATER = {"su_alti", "su_ustu"}


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u else 0


def select(n_torq: int, n_neg: int, seed: int) -> list[dict]:
    audit = list(csv.DictReader((META / "audit.csv").open(encoding="utf-8")))
    ds_groups = {a["dup_group"] for a in audit if a["split"] in ("train", "valid", "test")}
    in_ds = {a["path"] for a in audit if a["split"] == "inat" and a["dup_group"] in ds_groups}
    pre = json.loads((META / "inat_prelabels.json").read_text(encoding="utf-8"))
    photos = list(csv.DictReader((META / "inat_photos.csv").open(encoding="utf-8")))
    seen_obs, pool = set(), {"lago": [], "torq": [], "neg": []}
    for p in sorted(photos, key=lambda p: (p["observation_id"], p["photo_id"])):
        f = p["file"]
        if f in in_ds or p["observation_id"] in seen_obs or pre.get(f, {}).get("env_auto") not in WATER:
            continue
        seen_obs.add(p["observation_id"])
        key = "neg" if p["role"] == "negatif" else ("lago" if p["taxon"].startswith("Lagocephalus") else "torq")
        pool[key].append(p)
    rng = random.Random(seed)
    for k in pool:
        rng.shuffle(pool[k])
    chosen = pool["lago"] + pool["torq"][:n_torq] + pool["neg"][:n_neg]
    print(f"secilen: lago={len(pool['lago'])} torq={min(n_torq, len(pool['torq']))} neg={min(n_neg, len(pool['neg']))}")
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-torq", type=int, default=80)
    ap.add_argument("--n-neg", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--owl-thr", type=float, default=0.12)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    from torchvision.ops import nms

    chosen = select(args.n_torq, args.n_neg, args.seed)
    pre = json.loads((META / "inat_prelabels.json").read_text(encoding="utf-8"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    name = "google/owlv2-base-patch16-ensemble"
    proc = Owlv2Processor.from_pretrained(name)
    model = Owlv2ForObjectDetection.from_pretrained(name).to(dev).eval()
    queries = [["a photo of a fish", "a pufferfish"]]

    out = {}
    for k, p in enumerate(chosen):
        im = Image.open(ROOT / p["file"]).convert("RGB")
        w, h = im.size
        with torch.no_grad():
            inp = proc(text=queries, images=im, return_tensors="pt").to(dev)
            res = model(**inp)
        # OWLv2 goruntuyu kare olacak sekilde doldurur: hedef boyut max(w,h) kare
        s = max(w, h)
        r = proc.post_process_grounded_object_detection(res, threshold=args.owl_thr,
                                                        target_sizes=torch.tensor([[s, s]], device=dev))[0]
        boxes, scores = r["boxes"], r["scores"]
        keep = nms(boxes, scores, 0.5) if len(boxes) else []
        cand = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i].tolist()
            b = [max(x1, 0) / w, max(y1, 0) / h, min(x2, w) / w, min(y2, h) / h]
            if b[2] - b[0] > 0.01 and b[3] - b[1] > 0.01 and (b[2] - b[0]) * (b[3] - b[1]) < 0.95:
                cand.append([round(v, 4) for v in b] + ["owl", round(float(scores[i]), 3)])
        for yb in pre.get(p["file"], {}).get("boxes", []):
            if yb[4] >= 0.15 and all(iou(yb, c) < 0.6 for c in cand):
                cand.append([round(v, 4) for v in yb[:4]] + ["yolo", yb[4]])
        out[p["file"]] = {"role": p["role"], "taxon": p["taxon"], "obs_url": p["obs_url"], "boxes": cand}
        if k % 25 == 0:
            print(f"{k}/{len(chosen)}")
    (META / "gold_candidates.json").write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
    print("->", META / "gold_candidates.json")


if __name__ == "__main__":
    main()
