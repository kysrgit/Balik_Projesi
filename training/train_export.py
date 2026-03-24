"""
Deprecated training entrypoint.

The project now uses the standardized dataset pipeline:
    1. python training/data_prep.py
    2. python training/train_yolo.py
    3. python training/export_quantize.py
"""


def main():
    print("UYARI: training/train_export.py deprecated.")
    print("Standart pipeline icin su komutlari kullanin:")
    print("  python training/data_prep.py")
    print("  python training/train_yolo.py")
    print("  python training/export_quantize.py")


if __name__ == "__main__":
    main()
