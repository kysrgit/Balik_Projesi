import os
import site

#  Add only BIN directories (DLLs are in bin, not lib on Windows)
site_packages = site.getsitepackages()[0]
nvidia_bins = [
    os.path.join(site_packages, "nvidia", "cudnn", "bin"),
    os.path.join(site_packages, "nvidia", "cublas", "bin"),
    os.path.join(site_packages, "nvidia", "cuda_runtime", "bin"),
    os.path.join(site_packages, "nvidia", "cufft", "bin"),
    os.path.join(site_packages, "nvidia", "curand", "bin"),
    os.path.join(site_packages, "nvidia", "cusolver", "bin"),
    os.path.join(site_packages, "nvidia", "cusparse", "bin"),
    os.path.join(site_packages, "nvidia", "nv vtools_ext", "bin"),
]

print("Adding NVIDIA BIN paths to PATH:")
for bin_path in nvidia_bins:
    if os.path.exists(bin_path):
        os.add_dll_directory(bin_path)
        os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]
        print(f"  ✅ {bin_path}")

print("\nTesting ONNX Runtime CUDA:")
import onnxruntime as ort

try:
    session = ort.InferenceSession("yolo11m_fp16.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    print(f"✅ Active providers: {session.get_providers()}")
except Exception as e:
    print(f"❌ Failed: {str(e)[:200]}")
