import os
import site

# Add NVIDIA paths
site_packages = site.getsitepackages()[0]
nvidia_libs = [
    os.path.join(site_packages, "nvidia", "cudnn", "bin"),
    os.path.join(site_packages, "nvidia", "cudnn", "lib"),
    os.path.join(site_packages, "nvidia", "cublas", "bin"),
    os.path.join(site_packages, "nvidia", "cublas", "lib"),
    os.path.join(site_packages, "nvidia", "cuda_runtime", "bin"),
]

print("Checking NVIDIA library paths:")
for lib_path in nvidia_libs:
    exists = os.path.exists(lib_path)
    print(f"  {'✅' if exists else '❌'} {lib_path}")
    if exists:
        os.add_dll_directory(lib_path)
        os.environ["PATH"] = lib_path + os.pathsep + os.environ["PATH"]

print("\nTesting ONNX Runtime CUDA provider:")
import onnxruntime as ort

try:
    print(f"Available providers: {ort.get_available_providers()}")
    
    # Try to create a CUDA session
    session = ort.InferenceSession("yolo11m_fp16.onnx", providers=['CUDAExecutionProvider'])
    print(f"✅ SUCCESS! Active providers: {session.get_providers()}")
except Exception as e:
    print(f"❌ FAILED: {e}")
