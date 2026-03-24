import numpy as np
from unittest.mock import MagicMock, patch

import app.main as main_mod


def test_run_starts_camera_thread(tmp_path):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    started_camera = MagicMock()
    started_camera.get_frame.side_effect = [frame, KeyboardInterrupt]

    detector = MagicMock()
    detector.detect.return_value = ([], [])

    with patch.object(main_mod, "CSV_LOG_FILE", str(tmp_path / "detections.csv")), \
         patch("app.main.Camera") as mock_camera, \
         patch("app.main.Detector", return_value=detector), \
         patch("app.main.gpio.init"), \
         patch("app.main.gpio.off"), \
         patch("app.main.gpio.on"), \
         patch("app.main.cv2.destroyAllWindows"):
        mock_camera.return_value.start.return_value = started_camera

        main_mod.run(use_clahe=True, show_gui=False)

    mock_camera.return_value.start.assert_called_once()
    started_camera.release.assert_called_once()
