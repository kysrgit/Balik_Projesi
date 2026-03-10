import re

with open("app/dashboard/stream.py", "r") as f:
    content = f.read()

replacement = """
# Cache for base64 encoded frames to avoid re-encoding the same frame multiple times.
# Key: (stream_type, quality), Value: (frame_id, base64_string)
_b64_cache = {}

def get_base64_frame(buffer, stream_type='detection', quality=50):
    \"\"\"WebSocket üzerinden gönderim için frame'i base64 string yapar\"\"\"
    global _b64_cache

    frame = buffer.get(stream_type)
    if frame is None:
        return None

    # We can use the object identity (id(frame)) as cache key.
    # Because FrameBuffer.update makes a .copy(), a new frame will have a new id.
    frame_id = id(frame)
    cache_key = (stream_type, quality)

    if cache_key in _b64_cache:
        cached_id, cached_b64 = _b64_cache[cache_key]
        if cached_id == frame_id:
            return cached_b64

    if stream_type in ['raw', 'clahe']:
        frame = cv2.resize(frame, (320, 240))

    _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    result = base64.b64encode(jpg).decode('utf-8')

    _b64_cache[cache_key] = (frame_id, result)
    return result
"""

old_func_pattern = r"def get_base64_frame\(.*?decode\('utf-8'\)"
new_content = re.sub(old_func_pattern, replacement.strip(), content, flags=re.DOTALL)

with open("app/dashboard/stream.py", "w") as f:
    f.write(new_content)
