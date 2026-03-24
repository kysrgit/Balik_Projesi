import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2


CSV_HEADER = ["Timestamp", "Date", "Time", "Confidence", "BBox_X1", "BBox_Y1", "BBox_X2", "BBox_Y2"]
DetectionBox = Tuple[int, int, int, int]


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_csv_header(csv_path: str | Path) -> Path:
    path = Path(csv_path)
    ensure_directory(path.parent)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(CSV_HEADER)
    return path


def build_detection_rows(
    boxes: Sequence[DetectionBox],
    confs: Sequence[float],
    now_dt: datetime | None = None,
) -> List[List[object]]:
    now_dt = now_dt or datetime.now()
    ts = now_dt.strftime("%H%M%S_%f")
    rows: List[List[object]] = []
    for (x1, y1, x2, y2), conf in zip(boxes, confs):
        rows.append(
            [
                ts,
                now_dt.strftime("%Y-%m-%d"),
                now_dt.strftime("%H:%M:%S"),
                round(float(conf), 4),
                int(x1),
                int(y1),
                int(x2),
                int(y2),
            ]
        )
    return rows


def append_detection_rows(csv_path: str | Path, rows: Iterable[Sequence[object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path = ensure_csv_header(csv_path)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def prune_old_files(directory: str | Path, max_files: int, pattern: str = "*.jpg") -> None:
    if max_files <= 0:
        return
    folder = ensure_directory(directory)
    files = sorted(
        [path for path in folder.glob(pattern) if path.is_file()],
        key=lambda item: item.stat().st_mtime,
    )
    while len(files) > max_files:
        oldest = files.pop(0)
        try:
            oldest.unlink()
        except OSError:
            break


def save_image_with_retention(
    image,
    directory: str | Path,
    filename: str,
    max_files: int,
    pattern: str = "*.jpg",
) -> Path:
    folder = ensure_directory(directory)
    path = folder / filename
    cv2.imwrite(str(path), image)
    prune_old_files(folder, max_files=max_files, pattern=pattern)
    return path


def save_thumbnail(
    frame,
    box: DetectionBox,
    directory: str | Path,
    filename: str,
    max_files: int,
    size: Tuple[int, int] = (100, 100),
    padding: int = 10,
) -> Path | None:
    x1, y1, x2, y2 = box
    thumb = frame[max(0, y1 - padding): y2 + padding, max(0, x1 - padding): x2 + padding]
    if thumb.size == 0:
        return None
    resized = cv2.resize(thumb, size)
    return save_image_with_retention(resized, directory, filename, max_files=max_files)
