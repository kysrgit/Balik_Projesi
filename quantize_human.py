from onnxruntime.quantization import quantize_dynamic, QuantType
import os

input_model = "yolo11n.onnx"
output_model = "app/yolo11n_int8.onnx"

print(f"Quantizing {input_model} to INT8...")

quantize_dynamic(
    input_model,
    output_model,
    weight_type=QuantType.QUInt8
)

print(f"Quantization complete. Saved to {output_model}")
