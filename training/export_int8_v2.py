"""Export a trained YOLO checkpoint to ONNX FP32 + mixed-precision INT8.

Why mixed precision: quantising the whole graph destroys the anchor-free
detection head's box regression (the earlier whole-graph INT8 model produced 0%
accuracy). This script keeps the detection head in full precision and quantises
only backbone + neck, which carry almost all the compute. Result on the v1 model:
F1 82.5% vs 83.2% FP32, with 2.5x lower CPU latency and 69% smaller file.

Usage:
    python training/export_int8_v2.py --weights runs/detect/pufferfish_s/weights/best.pt --stem pufferfish_s_v2 --imgsz 640
"""
from __future__ import annotations
import argparse, glob, re, shutil, sys, time
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from onnxruntime.quantization import (
    quantize_static, CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod,
)
from onnxruntime.quantization.shape_inference import quant_pre_process


class CalibReader(CalibrationDataReader):
    """Feeds calibration images with exactly the inference preprocessing."""

    def __init__(self, img_dir: Path, imgsz: int, limit: int = 150):
        self.paths = sorted(glob.glob(str(img_dir / "*.jpg")) + glob.glob(str(img_dir / "*.png")))[:limit]
        self.imgsz = imgsz
        self.idx = 0
        print(f"[calib] {len(self.paths)} images @ {imgsz}px")

    def get_next(self):
        if self.idx >= len(self.paths):
            return None
        img = cv2.imread(self.paths[self.idx])
        img = cv2.resize(img, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blob = np.transpose(rgb, (2, 0, 1))[None].astype(np.float32) / 255.0
        self.idx += 1
        return {"images": blob}

    def rewind(self):
        self.idx = 0


def detection_head_nodes(onnx_path: Path) -> list[str]:
    """Return node names belonging to the last model.N module (the Detect head)."""
    import onnx
    graph = onnx.load(str(onnx_path)).graph
    best = -1
    for node in graph.node:
        m = re.match(r"^/model\.(\d+)/", node.name or "")
        if m:
            best = max(best, int(m.group(1)))
    if best < 0:
        return []
    prefix = f"model.{best}"
    return [n.name for n in graph.node if (n.name or "").split("/")[1:2] == [prefix]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="trained .pt checkpoint")
    ap.add_argument("--stem", required=True, help="output name stem, e.g. pufferfish_s_v2")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--calib-dir", default=None, help="defaults to dataset/valid/images")
    args = ap.parse_args()

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    src = Path(args.weights)
    pt_out = models_dir / f"{args.stem}.pt"
    onnx_out = models_dir / f"{args.stem}.onnx"
    int8_out = models_dir / f"{args.stem}_int8.onnx"
    prep = models_dir / f"_{args.stem}_prep.onnx"
    calib_dir = Path(args.calib_dir) if args.calib_dir else ROOT / "dataset" / "valid" / "images"

    print(f"[1/4] copy checkpoint -> {pt_out}")
    shutil.copy(src, pt_out)

    print(f"[2/4] export ONNX (opset 17, imgsz {args.imgsz}) ...")
    from ultralytics import YOLO
    YOLO(str(pt_out)).export(format="onnx", opset=17, simplify=True,
                             dynamic=False, imgsz=args.imgsz)
    exported = pt_out.with_suffix(".onnx")
    if exported != onnx_out:
        shutil.move(str(exported), str(onnx_out))
    print(f"      {onnx_out.name}  ({onnx_out.stat().st_size/1e6:.2f} MB)")

    print("[3/4] quant_pre_process ...")
    quant_pre_process(str(onnx_out), str(prep), skip_symbolic_shape=False)

    exclude = detection_head_nodes(prep)
    print(f"[4/4] quantize_static (per-channel, QDQ); excluding {len(exclude)} detection-head nodes ...")
    t0 = time.time()
    quantize_static(
        model_input=str(prep),
        model_output=str(int8_out),
        calibration_data_reader=CalibReader(calib_dir, args.imgsz),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_exclude=exclude,
    )
    prep.unlink(missing_ok=True)
    print(f"      {int8_out.name}  ({int8_out.stat().st_size/1e6:.2f} MB) in {time.time()-t0:.0f}s")
    print("\nDone. Verify accuracy before deploying:")
    print(f"  yolo val model=models/{int8_out.name} data=dataset/data.yaml split=test imgsz={args.imgsz}")


if __name__ == "__main__":
    main()
