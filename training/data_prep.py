import os
import shutil
import random
from pathlib import Path
import yaml

# --- CONFIGURATION ---
# Source directory containing raw images (User provided path)
SOURCE_DIR = r"c:\AI\Balik_Projesi_Antigravity\balon_baligi_fotograflari"

# Destination YOLO structure
BASE_DIR = Path(__file__).parent.parent / "dataset"
IMG_DIR = BASE_DIR / "images"
LBL_DIR = BASE_DIR / "labels"

# Split ratio
TRAIN_RATIO = 0.8

# Supported extensions
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def setup_directories():
    """Creates the YOLO directory structure."""
    if BASE_DIR.exists():
        print(f"[INFO] Dataset directory {BASE_DIR} already exists. Merging/Updating...")
    
    for split in ['train', 'val']:
        (IMG_DIR / split).mkdir(parents=True, exist_ok=True)
        (LBL_DIR / split).mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Dataset structure ready at {BASE_DIR}")

def create_yaml_config():
    """Creates the data.yaml file for YOLO training."""
    # YOLO expects paths relative to the execution dir or absolute paths.
    # We will use absolute paths to be safe.
    data = {
        'path': str(BASE_DIR.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 2,
        'names': ['pufferfish', 'fish'] # 0: pufferfish, 1: general fish
    }
    
    yaml_path = BASE_DIR / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"[INFO] Created YOLO config at {yaml_path}")

def process_images():
    """Scans source, splits, and copies images."""
    source_path = Path(SOURCE_DIR)
    
    if not source_path.exists():
        print(f"[ERROR] Source directory not found: {SOURCE_DIR}")
        print("Please ensure the 'balon_baligi_fotograflari' folder exists.")
        return

    # Gather all valid images recursively
    images = []
    for ext in EXTENSIONS:
        images.extend(source_path.rglob(f"*{ext}"))
        images.extend(source_path.rglob(f"*{ext.upper()}"))
    
    if not images:
        print("[WARNING] No images found in source directory!")
        return

    # Shuffle and split
    random.shuffle(images)
    split_idx = int(len(images) * TRAIN_RATIO)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    print(f"[INFO] Found {len(images)} images. Split: {len(train_imgs)} Train, {len(val_imgs)} Val.")

    def copy_files(file_list, split_name):
        dest_img_path = IMG_DIR / split_name
        count = 0
        for img in file_list:
            dest_file = dest_img_path / img.name
            if not dest_file.exists():
                shutil.copy2(img, dest_file)
                count += 1
        print(f"[INFO] Copied {count} new images to {split_name}")

    copy_files(train_imgs, 'train')
    copy_files(val_imgs, 'val')
    
    print("\n" + "="*50)
    print("DATA PREPARATION COMPLETE")
    print("="*50)
    print("[CRITICAL ACTION REQUIRED]")
    print("1. The images are now in: dataset/images/train and dataset/images/val")
    print("2. You MUST now use a tool like 'LabelImg' or 'Roboflow' to annotate these images.")
    print("3. Save the annotation .txt files into:")
    print(f"   - {LBL_DIR}/train")
    print(f"   - {LBL_DIR}/val")
    print("4. Use the following class IDs:")
    print("   0: pufferfish")
    print("   1: fish")
    print("="*50)

if __name__ == "__main__":
    setup_directories()
    create_yaml_config()
    process_images()
