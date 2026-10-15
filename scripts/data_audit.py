"""Veri denetimi (Faz 1): kopyalar, setler arasi sizinti, etiket istatistikleri, ortam onerisi.

Her goruntu icin:
  - pHash (64 bit) -> gorsel kopya / neredeyse ayni goruntu gruplari (Hamming <= --dup-dist)
  - etiket: ornek sayisi, en kucuk kutu alani, poligon mu kutu mu
  - CLIP ile ortam onerisi: su_alti / su_ustu / kara (kesin degil; inceleme arayuzunden dogrulanir)

Ciktilar:
  dataset_meta/audit.csv            goruntu basina satir (git'te tutulur, kucuk)
  dataset_meta/audit_summary.json   ozet sayilar

Kullanim (GPU ortami, open_clip + imagehash gerekir):
    .venv_gpu/Scripts/python.exe scripts/data_audit.py
"""
from __future__ import annotations
import argparse, csv, json, re
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
RAW = ROOT / "balon_baligi_fotograflari"
OUT = ROOT / "dataset_meta"
SPLITS = ("train", "valid", "test")
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# CLIP istemleri: her ortam icin birkac ifade, skorlar ortalanir
ENV_PROMPTS = {
    "su_alti": ["an underwater photo of a fish swimming in the sea",
                "a photo taken underwater, blue or green water, seabed",
                "a scuba diving photo of a fish near rocks and seagrass"],
    "su_ustu": ["a photo of a fish seen from above the water surface",
                "a fish in a bucket of water or an aquarium tank viewed from above"],
    "kara": ["a photo of a dead fish lying on the ground, a boat deck or a table",
             "a fisherman holding a caught fish in his hand",
             "a fish on a fish market counter or in a net out of water"],
}


def source_name(name: str) -> str:
    """Roboflow dosya adindan ozgun kaynak adini cikar: 'fotograf_12_jpg.rf.abc.jpg' -> 'fotograf_12'."""
    m = re.match(r"(.+?)_(?:jpe?g|png)\.rf\.", name)
    return m.group(1) if m else Path(name).stem


def label_stats(path: Path) -> dict:
    if not path.exists():
        return {"instances": 0, "min_area": "", "format": "yok"}
    areas, fmt = [], set()
    for line in path.read_text(encoding="utf-8").splitlines():
        v = list(map(float, line.split()[1:]))
        if len(v) == 4:
            areas.append(v[2] * v[3]); fmt.add("kutu")
        elif len(v) > 4:
            xs, ys = v[0::2], v[1::2]
            areas.append((max(xs) - min(xs)) * (max(ys) - min(ys))); fmt.add("poligon")
    return {"instances": len(areas), "min_area": round(min(areas), 5) if areas else "",
            "format": "+".join(sorted(fmt)) or "bos"}


def collect() -> list[dict]:
    rows = []
    for split in SPLITS:
        for p in sorted((DATASET / split / "images").iterdir()):
            if p.suffix.lower() not in IMG_EXT:
                continue
            r = {"split": split, "file": p.name, "path": p.relative_to(ROOT).as_posix(),
                 "source": source_name(p.name)}
            r.update(label_stats(DATASET / split / "labels" / f"{p.stem}.txt"))
            rows.append(r)
    for p in sorted(RAW.rglob("*")) if RAW.exists() else []:
        if p.suffix.lower() in IMG_EXT:
            rows.append({"split": "ham", "file": p.name, "path": p.relative_to(ROOT).as_posix(),
                         "source": p.stem, "instances": "", "min_area": "", "format": ""})
    return rows


def add_hashes(rows: list[dict]) -> None:
    for r in rows:
        with Image.open(ROOT / r["path"]) as im:
            r["w"], r["h"] = im.size
            r["phash"] = str(imagehash.phash(im.convert("RGB")))


def dup_groups(rows: list[dict], max_dist: int) -> None:
    """Union-find ile pHash mesafesi <= max_dist olan goruntuleri grupla."""
    hashes = [imagehash.hex_to_hash(r["phash"]) for r in rows]
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if hashes[i] - hashes[j] <= max_dist:
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    for gid, (_, members) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1]))):
        for i in members:
            rows[i]["dup_group"] = gid
            rows[i]["dup_size"] = len(members)


def add_env(rows: list[dict], batch: int = 64) -> None:
    import open_clip, torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k")
    tok = open_clip.get_tokenizer("ViT-B-16")
    model = model.to(device).eval()
    labels, prompts = [], []
    for env, ps in ENV_PROMPTS.items():
        labels += [env] * len(ps); prompts += ps
    with torch.no_grad():
        t = model.encode_text(tok(prompts).to(device)); t /= t.norm(dim=-1, keepdim=True)
        for k in range(0, len(rows), batch):
            chunk = rows[k:k + batch]
            ims = torch.stack([preprocess(Image.open(ROOT / r["path"]).convert("RGB")) for r in chunk]).to(device)
            f = model.encode_image(ims); f /= f.norm(dim=-1, keepdim=True)
            probs = (100 * f @ t.T).softmax(dim=-1).cpu()
            for r, pr in zip(chunk, probs):
                score = {env: float(sum(pr[i] for i, l in enumerate(labels) if l == env)) for env in ENV_PROMPTS}
                best = max(score, key=score.get)
                r["env_auto"] = best
                r["env_conf"] = round(score[best], 3)


def summarize(rows: list[dict]) -> dict:
    ds = [r for r in rows if r["split"] != "ham"]
    s = {"images": Counter(r["split"] for r in rows)}
    s["env_auto_by_split"] = {sp: dict(Counter(r["env_auto"] for r in ds if r["split"] == sp)) for sp in SPLITS}
    s["env_auto_low_conf(<0.6)"] = sum(1 for r in ds if r["env_conf"] < 0.6)
    # sizinti: ayni pHash grubunda farkli split'ler
    by_group = defaultdict(set)
    for r in ds:
        by_group[r["dup_group"]].add(r["split"])
    leak = Counter()
    for splits in by_group.values():
        for a in SPLITS:
            for b in SPLITS:
                if a < b and a in splits and b in splits:
                    leak[f"{a}&{b}"] += 1
    s["leak_groups_by_phash"] = dict(leak)
    s["dup_groups_within_test"] = sum(1 for g, sp in by_group.items()
                                      if sum(1 for r in ds if r["dup_group"] == g and r["split"] == "test") > 1)
    raw_groups = {r["dup_group"] for r in rows if r["split"] == "ham"}
    ds_groups = {r["dup_group"] for r in ds}
    s["raw_not_in_dataset"] = sum(1 for r in rows if r["split"] == "ham" and r["dup_group"] not in ds_groups)
    s["raw_in_dataset"] = sum(1 for r in rows if r["split"] == "ham" and r["dup_group"] in ds_groups)
    s["dataset_groups_not_in_raw"] = len(ds_groups - raw_groups)
    s["label_format"] = dict(Counter(r["format"] for r in ds))
    s["negatives_by_split"] = {sp: sum(1 for r in ds if r["split"] == sp and r["instances"] == 0) for sp in SPLITS}
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dup-dist", type=int, default=6, help="pHash Hamming esigi (0-64)")
    ap.add_argument("--no-clip", action="store_true")
    args = ap.parse_args()
    rows = collect()
    print(f"{len(rows)} goruntu toplandi, hash hesaplaniyor...")
    add_hashes(rows)
    dup_groups(rows, args.dup_dist)
    if args.no_clip:
        for r in rows:
            r["env_auto"], r["env_conf"] = "", 0.0
    else:
        print("CLIP ortam onerisi hesaplaniyor...")
        add_env(rows)
    OUT.mkdir(exist_ok=True)
    fields = ["split", "file", "path", "source", "w", "h", "instances", "min_area", "format",
              "phash", "dup_group", "dup_size", "env_auto", "env_conf"]
    with (OUT / "audit.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    summary = summarize(rows)
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
