"""
Tests for get_stats in app/dashboard/server.py
"""
import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch


def _load_get_stats():
    mock_modules = {
        "cv2": MagicMock(),
        "flask": MagicMock(),
        "flask_socketio": MagicMock(),
        "app.core": MagicMock(),
        "app.utils": MagicMock(),
        "app.utils.image": MagicMock(),
        "app.dashboard.stream": MagicMock(),
        "app.core.gps": MagicMock(),
        "app.core.detection_store": MagicMock(),
        "app.db.spatial": MagicMock(),
        "app.export": MagicMock(),
    }
    eventlet_mock = MagicMock()
    tpool_mock = MagicMock()
    tpool_mock.execute.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)
    eventlet_mock.tpool = tpool_mock
    mock_modules["eventlet"] = eventlet_mock
    mock_modules["eventlet.tpool"] = tpool_mock

    with patch.dict(sys.modules, mock_modules, clear=False), \
         patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open()):
        server = importlib.import_module("app.dashboard.server")
        get_stats = server.get_stats

    sys.modules.pop("app.dashboard.server", None)
    return get_stats


get_stats = _load_get_stats()


class TestGetStats(unittest.TestCase):
    def test_get_stats_success(self):
        # glob mock'u: gercek Pi'de /sys/class/hwmon/*/fan1_input bulunur ve
        # read_text mock'u fan RPM'ine de sizar.
        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "glob", return_value=iter([])), \
             patch.object(Path, "read_text", return_value="45000\n"):
            stats = get_stats()
            self.assertEqual(stats["cpu_temp"], 45.0)
            self.assertFalse(stats["throttled"])
            self.assertIsNone(stats["fan_rpm"])

    def test_get_stats_file_not_found(self):
        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "read_text", side_effect=FileNotFoundError):
            stats = get_stats()
            self.assertEqual(stats["cpu_temp"], 0)
            self.assertFalse(stats["throttled"])
            self.assertIsNone(stats["fan_rpm"])

    def test_get_stats_permission_error(self):
        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "read_text", side_effect=PermissionError):
            stats = get_stats()
            self.assertEqual(stats["cpu_temp"], 0)

    def test_get_stats_invalid_data(self):
        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "read_text", return_value="invalid_temp"):
            stats = get_stats()
            self.assertEqual(stats["cpu_temp"], 0)
