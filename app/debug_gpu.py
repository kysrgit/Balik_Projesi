import onnxruntime as ort
import os

print("--- ONNX Runtime GPU Diagnostic ---")
print(f"ONNX Runtime Version: {ort.__version__}")
print(f"Available Providers: {ort.get_available_providers()}")

try:
    # Try to initialize a session with CUDA
    # We need a dummy model or just check if we can create a session options with CUDA
    print("\nAttempting to load CUDA provider...")
    providers = ['CUDAExecutionProvider']
    # Create a dummy session or just check build info
    print(f"Build Info: {ort.get_build_info()}")
    
    # Check if CUDA is actually usable
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        print("CUDAProvider is listed in available providers.")
    else:
        print("CUDAProvider is NOT listed. This usually means CUDA/cuDNN DLLs are missing or version mismatch.")
        
except Exception as e:
    print(f"Error: {e}")

print("\n--- End Diagnostic ---")
