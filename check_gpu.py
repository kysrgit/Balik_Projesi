import torch

print("=== GPU Diagnostic ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("WARNING: CUDA is NOT available!")
    print("This means PyTorch cannot see your NVIDIA GPU.")
    print("Possible reasons:")
    print("  1. CUDA toolkit not installed")
    print("  2. PyTorch CPU-only version installed")
    print("  3. GPU drivers outdated")
