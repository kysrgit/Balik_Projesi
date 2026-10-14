"""
Şekil 5 (Fig 5) Grafik Üretim Betiği.
YOLO11s modellerinin (PyTorch, ONNX FP32, ONNX INT8, GPU) çıkarım gecikmesi (ms),
model disk boyutu (MB) ve test mAP@50 doğruluk karşılaştırmasını çizer.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "archive" / "report_assets" / "fig5_modelcompare_en.png"

# Veriler (CPU ve GPU ampirik sonuçları)
formats = ["PyTorch (.pt)", "ONNX FP32", "ONNX INT8 (Proposed)", "PyTorch (GPU)"]
latencies = [118.48, 98.30, 56.89, 8.80]       # ms per frame
sizes = [19.2, 37.9, 12.5, 19.2]               # MB
test_maps = [75.24, 75.16, 74.52, 75.24]       # test mAP@50 (%)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax1 = plt.subplots(figsize=(9, 5.5), dpi=300)

colors = ['#2b5c8f', '#4682b4', '#2ca02c', '#d62728']

# Çubuk grafik (Gecikme ms)
bars = ax1.bar(formats, latencies, color=colors, alpha=0.85, width=0.45, edgecolor='black', linewidth=0.8)
ax1.set_ylabel('Inference Latency (ms / frame)', fontsize=11, fontweight='bold', color='#333333')
ax1.set_ylim(0, 140)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# Değer etiketlerini barların üstüne ekle
for bar, lat, size in zip(bars, latencies, sizes):
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 3.0,
             f"{lat:.1f} ms\n({size:.1f} MB)",
             ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#111111')

# İkinci eksen (test mAP@50)
ax2 = ax1.twinx()
ax2.plot(formats, test_maps, color='#e377c2', marker='o', linewidth=2.5, markersize=8, label='Test mAP@50 (%)')
ax2.set_ylabel('Test mAP@50 (%)', fontsize=11, fontweight='bold', color='#e377c2')
ax2.set_ylim(65, 80)
ax2.grid(False)

for i, txt in enumerate(test_maps):
    ax2.annotate(f"{txt:.2f}%", (formats[i], test_maps[i]),
                 textcoords="offset points", xytext=(0,10), ha='center',
                 fontsize=9.5, fontweight='bold', color='#b5179e')

plt.title('Figure 5: Model Format Comparison — Latency, Model Size, and Test Accuracy (YOLO11s)',
          fontsize=12, fontweight='bold', pad=15)
fig.tight_layout()

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PATH, dpi=300, bbox_inches='tight')
print(f"Görsel başarıyla kaydedildi -> {OUT_PATH}")
