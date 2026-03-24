import time

import numpy as np

from app.core.detection_store import prune_old_files, save_thumbnail


def test_prune_old_files_keeps_latest(tmp_path):
    for index in range(4):
        path = tmp_path / f"img_{index}.jpg"
        path.write_bytes(b"test")
        time.sleep(0.01)

    prune_old_files(tmp_path, max_files=3)

    remaining = sorted(path.name for path in tmp_path.glob("*.jpg"))
    assert remaining == ["img_1.jpg", "img_2.jpg", "img_3.jpg"]


def test_save_thumbnail_applies_retention(tmp_path):
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    save_thumbnail(frame, (10, 10, 100, 100), tmp_path, "one.jpg", max_files=2)
    save_thumbnail(frame, (10, 10, 100, 100), tmp_path, "two.jpg", max_files=2)
    save_thumbnail(frame, (10, 10, 100, 100), tmp_path, "three.jpg", max_files=2)

    remaining = sorted(path.name for path in tmp_path.glob("*.jpg"))
    assert remaining == ["three.jpg", "two.jpg"]
