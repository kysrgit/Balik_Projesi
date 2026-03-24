# Dataset hazirlama - standart train/valid/test YOLO formati
import random
import shutil
from pathlib import Path

import yaml


SOURCE_DIR = Path(__file__).parent.parent / "balon_baligi_fotograflari"
BASE_DIR = Path(__file__).parent.parent / "dataset"
SPLITS = {
    "train": 0.80,
    "valid": 0.10,
    "test": 0.10,
}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = ["pufferfish"]


def _split_directories(split_name: str) -> tuple[Path, Path]:
    return BASE_DIR / split_name / "images", BASE_DIR / split_name / "labels"


def setup_dirs():
    for split in SPLITS:
        img_dir, lbl_dir = _split_directories(split)
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
    print(f"Klasorler hazir: {BASE_DIR}")


def create_yaml():
    data = {
        "path": str(BASE_DIR.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    with (BASE_DIR / "data.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)
    print("data.yaml olusturuldu")


def collect_images():
    images = []
    for extension in EXTENSIONS:
        images.extend(SOURCE_DIR.rglob(f"*{extension}"))
        images.extend(SOURCE_DIR.rglob(f"*{extension.upper()}"))
    return images


def copy_images():
    if not SOURCE_DIR.exists():
        print(f"Kaynak bulunamadi: {SOURCE_DIR}")
        return

    images = collect_images()
    if not images:
        print("Gorsel yok!")
        return

    random.shuffle(images)
    total = len(images)
    train_cutoff = int(total * SPLITS["train"])
    valid_cutoff = train_cutoff + int(total * SPLITS["valid"])

    for index, image_path in enumerate(images):
        if index < train_cutoff:
            split = "train"
        elif index < valid_cutoff:
            split = "valid"
        else:
            split = "test"

        image_dir, _ = _split_directories(split)
        target = image_dir / image_path.name
        if not target.exists():
            shutil.copy2(image_path, target)

    print(
        f"{total} gorsel kopyalandi "
        f"(train: {train_cutoff}, valid: {valid_cutoff - train_cutoff}, test: {total - valid_cutoff})"
    )
    print("\nEtiketleme icin label dosyalarini dataset/<split>/labels klasorlerine koyun.")


if __name__ == "__main__":
    setup_dirs()
    create_yaml()
    copy_images()
