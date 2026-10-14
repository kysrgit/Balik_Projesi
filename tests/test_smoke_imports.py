import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "app.main",
        "app.dashboard.server",
        "training.data_prep",
        "training.train_v2_improved",
        "training.export_int8_v2",
    ],
)
def test_module_imports_in_fresh_process(module_name):
    result = subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
