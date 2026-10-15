"""iNaturalist'ten acik lisansli gozlem fotograflarini indirir (Faz 1, ADR-001).

Yalnizca arastirma kalitesindeki gozlemler ve acik lisansli fotograflar alinir.
Her fotografin lisansi, fotografcisi ve gozlem baglantisi kaydedilir.

Ciktilar:
  dataset_inat/<rol>/<takson>/<gozlem>_<foto>.jpg   (git disinda)
  dataset_meta/inat_photos.csv                      (meta veri, git'te)

Kullanim:
    python scripts/fetch_inat.py                 # varsayilan takson listesi
    python scripts/fetch_inat.py --max-neg-obs 150
"""
from __future__ import annotations
import argparse, csv, json, time, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dataset_inat"
META = ROOT / "dataset_meta" / "inat_photos.csv"
API = "https://api.inaturalist.org/v1"
LICENSES = "cc0,cc-by,cc-by-nc,cc-by-sa,cc-by-nc-sa"
UA = {"User-Agent": "Balik_Projesi research data fetch (github.com/kysrgit/Balik_Projesi)"}

# rol -> takson listesi (ADR-001)
TAXA = {
    "pozitif": ["Lagocephalus sceleratus", "Lagocephalus suezensis", "Lagocephalus guentheri",
                "Torquigener flavimaculosus"],
    "negatif": ["Siganus luridus", "Siganus rivulatus", "Sparisoma cretense"],
}
FIELDS = ["role", "taxon", "observation_id", "photo_id", "file", "license", "attribution",
          "observer", "observed_on", "place", "lat", "lon", "obs_url", "photo_url"]


def get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def taxon_id(name: str) -> int:
    q = urllib.parse.urlencode({"q": name, "rank": "species", "per_page": 1})
    res = get_json(f"{API}/taxa?{q}")["results"]
    if not res or res[0]["name"] != name:
        raise SystemExit(f"Takson bulunamadi: {name}")
    return res[0]["id"]


def observations(tid: int, limit: int):
    page, got = 1, 0
    while got < limit:
        q = urllib.parse.urlencode({"taxon_id": tid, "photos": "true", "quality_grade": "research",
                                    "photo_license": LICENSES, "per_page": 200, "page": page,
                                    "order_by": "id", "order": "asc"})
        res = get_json(f"{API}/observations?{q}")["results"]
        if not res:
            return
        for o in res:
            yield o
            got += 1
            if got >= limit:
                return
        page += 1
        time.sleep(1.0)  # API: saniyede ~1 istek


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pos-obs", type=int, default=10_000)
    ap.add_argument("--max-neg-obs", type=int, default=150)
    ap.add_argument("--photos-per-obs", type=int, default=3)
    ap.add_argument("--size", default="large", choices=["medium", "large"])
    args = ap.parse_args()

    existing = {}
    if META.exists():
        with META.open(encoding="utf-8", newline="") as f:
            existing = {r["photo_id"]: r for r in csv.DictReader(f)}
    rows = dict(existing)
    allowed = set(LICENSES.split(","))

    for role, names in TAXA.items():
        limit = args.max_pos_obs if role == "pozitif" else args.max_neg_obs
        per_obs = args.photos_per_obs if role == "pozitif" else 1
        for name in names:
            tid = taxon_id(name)
            slug = name.replace(" ", "_")
            folder = OUT / role / slug
            folder.mkdir(parents=True, exist_ok=True)
            n_new = 0
            for o in observations(tid, limit):
                for ph in o.get("photos", [])[:per_obs]:
                    pid = str(ph["id"])
                    lic = (ph.get("license_code") or "").lower()
                    if lic not in allowed or pid in rows:
                        continue
                    url = ph["url"].replace("/square.", f"/{args.size}.")
                    fname = folder / f"{o['id']}_{pid}.jpg"
                    try:
                        req = urllib.request.Request(url, headers=UA)
                        with urllib.request.urlopen(req, timeout=60) as r:
                            data = r.read()
                        if not data.startswith(b"\xff\xd8"):  # yalnizca JPEG kabul et
                            continue
                        fname.write_bytes(data)
                    except Exception as e:
                        print(f"  atlandi {url}: {e}")
                        continue
                    loc = (o.get("location") or ",").split(",")
                    rows[pid] = {"role": role, "taxon": name, "observation_id": o["id"], "photo_id": pid,
                                 "file": fname.relative_to(ROOT).as_posix(), "license": lic,
                                 "attribution": ph.get("attribution", ""),
                                 "observer": (o.get("user") or {}).get("login", ""),
                                 "observed_on": o.get("observed_on") or "", "place": o.get("place_guess") or "",
                                 "lat": loc[0], "lon": loc[1] if len(loc) > 1 else "",
                                 "obs_url": f"https://www.inaturalist.org/observations/{o['id']}", "photo_url": url}
                    n_new += 1
                    time.sleep(0.2)
            print(f"{role:8s} {name:30s} +{n_new} foto")
            META.parent.mkdir(exist_ok=True)
            with META.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader(); w.writerows(rows.values())
    total_mb = sum(p.stat().st_size for p in OUT.rglob("*.jpg")) / 1e6
    print(f"Toplam {len(rows)} foto, {total_mb:.0f} MB -> {OUT}")


if __name__ == "__main__":
    main()
