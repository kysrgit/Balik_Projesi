"""iNaturalist goruntuleri icin on etiket (Faz 1).

Mevcut model dusuk esikle calistirilir; cikan kutular etiketleme arayuzunde
(/etiket) baslangic onerisi olarak gosterilir ve insan tarafindan duzeltilir.
CLIP ortam onerisi de ayni dosyaya eklenir.

Cikti: dataset_meta/inat_prelabels.json  {dosya: {"boxes": [[x1,y1,x2,y2,conf],...], "env_auto": .., "env_conf": ..}}

Kullanim (GPU ortami):
    .venv_gpu/Scripts/python.exe scripts/prelabel_inat.py --weights models/pufferfish_s_v2.pt
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
META = ROOT / "dataset_meta"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/pufferfish_s_v2.pt")
    ap.add_argument("--conf", type=float, default=0.20)
    args = ap.parse_args()

    from ultralytics import YOLO
    from PIL import Image
    import data_audit  # ayni CLIP istemlerini kullan

    with (META / "inat_photos.csv").open(encoding="utf-8", newline="") as f:
        photos = list(csv.DictReader(f))
    out_path = META / "inat_prelabels.json"
    out = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    todo = [p for p in photos if p["file"] not in out and (ROOT / p["file"]).exists()]
    print(f"{len(todo)} yeni goruntu")

    model = YOLO(str(ROOT / args.weights))
    rows = [{"path": p["file"]} for p in todo]
    data_audit.add_env(rows)
    for p, r in zip(todo, rows):
        res = model.predict(str(ROOT / p["file"]), conf=args.conf, imgsz=640, verbose=False)[0]
        h, w = res.orig_shape
        boxes = [[round(float(x1 / w), 5), round(float(y1 / h), 5), round(float(x2 / w), 5),
                  round(float(y2 / h), 5), round(float(c), 3)]
                 for (x1, y1, x2, y2), c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy())]
        with Image.open(ROOT / p["file"]) as im:
            size = im.size
        out[p["file"]] = {"boxes": boxes, "env_auto": r["env_auto"], "env_conf": r["env_conf"], "size": size}
    out_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print("->", out_path)


if __name__ == "__main__":
    main()
