import importlib
import os
import sys
import unittest
from unittest.mock import MagicMock, patch


def _load_config_module():
    mock_modules = {
        "app.core.camera": MagicMock(),
        "app.core.detector": MagicMock(),
    }
    with patch.dict(sys.modules, mock_modules, clear=False):
        from app.core import config
        return importlib.reload(config)


class TestConfigCORS(unittest.TestCase):
    def test_default_config(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ALLOWED_ORIGINS", None)
            config = _load_config_module()
            self.assertIsNone(config.ALLOWED_ORIGINS)

    def test_env_var_config_single(self):
        with patch.dict(os.environ, {"ALLOWED_ORIGINS": "http://localhost:3000"}, clear=False):
            config = _load_config_module()
            self.assertEqual(config.ALLOWED_ORIGINS, "http://localhost:3000")

    def test_env_var_config_multiple(self):
        with patch.dict(
            os.environ,
            {"ALLOWED_ORIGINS": "http://localhost:3000, http://example.com"},
            clear=False,
        ):
            config = _load_config_module()
            self.assertEqual(config.ALLOWED_ORIGINS, ["http://localhost:3000", "http://example.com"])


if __name__ == "__main__":
    unittest.main()
