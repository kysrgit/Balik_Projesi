# YOLO11m egitim scripti
from ultralytics import YOLO
from pathlib import Path

# --- Sualti Agresif Veri Artirimi (Albumentations Monkey-Patch) ---
# Bu bolum, egitim sirasinda rastgele sualti kosullari simulasyonu yapar.
# Ayrı bir LU2Net gibi derin ogrenme mimarisi kullanmak yerine sifir pre-process maliyetiyle model dayanikliligi artirilir.
try:
    from ultralytics.data.augment import Albumentations
    import albumentations as A

    def custom_albumentations_init(self, p=1.0):
        self.p = p
        self.transform = None
        try:
            T = [
                # 1. Kötü görüş (Bulanık Su / Partiküller) - Gauss ve Median Blur
                A.Blur(blur_limit=(3, 7), p=0.3),
                A.MedianBlur(blur_limit=5, p=0.1),
                
                # 2. Kontrast düşüklüğü ve ışık yansımaları
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=(-0.4, 0.0), p=0.4),
                
                # 3. Yeşil/Mavi su simülasyonu (Kırmızı ışık sualtında emilir, mavi ve yeşil baskın kalır)
                A.RGBShift(r_shift_limit=(-30, -10), g_shift_limit=(-10, 20), b_shift_limit=(10, 30), p=0.3),
                
                # 4. Ekstra renk titremeleri
                A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.2)
            ]
            self.transform = A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))
            print("[+] Özel Sualtı Albumentations Augmentasyonu AKTİF!")
        except Exception as e:
            print(f"[!] Albumentations ayarları yapılandırılamadı: {e}")

    Albumentations.__init__ = custom_albumentations_init
except ImportError:
    print("[-] Uyarı: 'albumentations' veya 'ultralytics' yüklü değil. Sualtı asıl veri artırımı atlandı.")
    print("[-] Kurulum için: pip install albumentations")
# -----------------------------------------------------------------

DATA_YAML = Path(__file__).parent.parent / "dataset" / "data.yaml"
OUTPUT_DIR = Path(__file__).parent.parent / "runs" / "detect"

# Egitim parametreleri
EPOCHS = 50
IMG_SIZE = 640
MODEL = 'yolo11m.pt'

def train():
    if not DATA_YAML.exists():
        print(f"data.yaml yok: {DATA_YAML}")
        print("Once data_prep.py calistir!")
        return
    
    print(f"Egitim basliyor: {MODEL}")
    model = YOLO(MODEL)
    
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        project=str(OUTPUT_DIR),
        name='pufferfish',
        
        # Standart YOLO parametreleri ile destekleyici sualti augmentasyonlari
        hsv_h=0.04,   # Hue (renk tonu)
        hsv_s=0.7,    # Saturation
        hsv_v=0.6,    # Value (parlaklik dengesizligi icin artirildi)
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,   # Karmaşıklığı artırmak (balıkları iç içe gizlemek)
        bgr=0.1,      # Channel swap ile rastgele renk şoku
        
        device=0,
        exist_ok=True
    )
    
    print(f"Bitti! Model: {OUTPUT_DIR}/pufferfish/weights/best.pt")

if __name__ == "__main__":
    train()
