# Dataset hazirlama - YOLO formati
import os
import shutil
import random
from pathlib import Path
import yaml

SOURCE_DIR = Path(__file__).parent.parent / "balon_baligi_fotograflari"
BASE_DIR = Path(__file__).parent.parent / "dataset"
IMG_DIR = BASE_DIR / "images"
LBL_DIR = BASE_DIR / "labels"

TRAIN_RATIO = 0.8
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def setup_dirs():
    for split in ['train', 'val']:
        (IMG_DIR / split).mkdir(parents=True, exist_ok=True)
        (LBL_DIR / split).mkdir(parents=True, exist_ok=True)
    print(f"Klasorler hazir: {BASE_DIR}")


def create_yaml():
    data = {
        'path': str(BASE_DIR.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 2,
        'names': ['pufferfish', 'fish']
    }
    with open(BASE_DIR / "data.yaml", 'w') as f:
        yaml.dump(data, f)
    print("data.yaml olusturuldu")


def copy_images():
    if not SOURCE_DIR.exists():
        print(f"Kaynak bulunamadi: {SOURCE_DIR}")
        return
    
    imgs = []
    for ext in EXTENSIONS:
        imgs.extend(SOURCE_DIR.rglob(f"*{ext}"))
        imgs.extend(SOURCE_DIR.rglob(f"*{ext.upper()}"))
    
    if not imgs:
        print("Gorsel yok!")
        return
    
    random.shuffle(imgs)
    split = int(len(imgs) * TRAIN_RATIO)
    
    for i, img in enumerate(imgs):
        dest = 'train' if i < split else 'val'
        target = IMG_DIR / dest / img.name
        if not target.exists():
            shutil.copy2(img, target)
    
    print(f"{len(imgs)} gorsel kopyalandi (train: {split}, val: {len(imgs)-split})")
    print("\nSimdi LabelImg ile etiketleyin!")


if __name__ == "__main__":
    setup_dirs()
    create_yaml()
    copy_images()
