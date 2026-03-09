import os
import sys
import unittest
from unittest.mock import MagicMock

# Setup sys.path to include project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Mock EVERYTHING to avoid imports of missing packages
mock_modules = [
    'cv2', 'eventlet', 'numpy', 'ultralytics', 'flask', 'flask_socketio',
    'app.core.detector', 'app.utils', 'app.dashboard.stream', 'app.core.camera',
    'app.core.gps', 'app.db.spatial', 'app.export', 'pyserial', 'pynmea2', 'onnxruntime'
]

for mod in mock_modules:
    sys.modules[mod] = MagicMock()

class TestConfigCORS(unittest.TestCase):
    def test_default_config(self):
        # Ensure env var is not set
        if 'ALLOWED_ORIGINS' in os.environ:
            del os.environ['ALLOWED_ORIGINS']

        # Load config to apply default
        from app.core import config
        import importlib
        importlib.reload(config)

        self.assertIsNone(config.ALLOWED_ORIGINS)

    def test_env_var_config_single(self):
        os.environ['ALLOWED_ORIGINS'] = 'http://localhost:3000'
        from app.core import config
        import importlib
        importlib.reload(config)

        self.assertEqual(config.ALLOWED_ORIGINS, 'http://localhost:3000')

    def test_env_var_config_multiple(self):
        os.environ['ALLOWED_ORIGINS'] = 'http://localhost:3000, http://example.com'
        from app.core import config
        import importlib
        importlib.reload(config)

        self.assertEqual(config.ALLOWED_ORIGINS, ['http://localhost:3000', 'http://example.com'])

if __name__ == '__main__':
    unittest.main()
