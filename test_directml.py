import onnxruntime as ort

print("Available Providers:")
print(ort.get_available_providers())

print("\nTrying to create DmlExecutionProvider session...")
try:
    # Create a simple test session
    providers = ['DmlExecutionProvider']
    print(f"Requesting providers: {providers}")
    
    # Test if DirectML can be initialized
    session = ort.InferenceSession("yolo11m_pufferfish.onnx", providers=providers)
    print(f"SUCCESS: Active providers: {session.get_providers()}")
except Exception as e:
    print(f"FAILED: {e}")
    print("\nFalling back to CPU...")
    session = ort.InferenceSession("yolo11m_pufferfish.onnx", providers=['CPUExecutionProvider'])
    print(f"CPU Active providers: {session.get_providers()}")
