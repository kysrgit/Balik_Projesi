"""
Tests for get_stats in app/dashboard/server.py
"""
import sys
import unittest
from unittest.mock import MagicMock, patch, mock_open

# 1. Mock all external and problematic dependencies *before* importing anything from the app.
# This avoids side effects like trying to import cv2, flask, or eventlet.
mock_modules = [
    'eventlet', 'cv2', 'flask', 'flask_socketio',
    'app.core', 'app.utils', 'app.dashboard.stream',
    'app.core.gps', 'app.db.spatial', 'app.export'
]
for mod in mock_modules:
    sys.modules[mod] = MagicMock()

# 2. Mock top-level side effects in app.dashboard.server (like CSV file creation)
# before importing the function we want to test.
with patch('os.path.exists', return_value=True), \
     patch('builtins.open', mock_open()):
    from app.dashboard.server import get_stats

class TestGetStats(unittest.TestCase):
    def test_get_stats_success(self):
        """Test get_stats when system file exists and contains valid data."""
        # Note: We patch builtins.open again inside the test to control the read data.
        with patch('builtins.open', mock_open(read_data='45000\n')):
            stats = get_stats()
            self.assertEqual(stats['cpu_temp'], 45.0)
            self.assertFalse(stats['throttled'])
            self.assertEqual(stats['fan_rpm'], 0)

    def test_get_stats_file_not_found(self):
        """Test get_stats when system file does not exist (e.g., non-Raspberry Pi environment)."""
        with patch('builtins.open', side_effect=FileNotFoundError):
            stats = get_stats()
            # Should return default values
            self.assertEqual(stats['cpu_temp'], 0)
            self.assertFalse(stats['throttled'])
            self.assertEqual(stats['fan_rpm'], 0)

    def test_get_stats_permission_error(self):
        """Test get_stats when system file cannot be accessed due to permissions."""
        with patch('builtins.open', side_effect=PermissionError):
            stats = get_stats()
            self.assertEqual(stats['cpu_temp'], 0)

    def test_get_stats_invalid_data(self):
        """Test get_stats when system file contains non-numeric data."""
        with patch('builtins.open', mock_open(read_data='invalid_temp')):
            stats = get_stats()
            self.assertEqual(stats['cpu_temp'], 0)
