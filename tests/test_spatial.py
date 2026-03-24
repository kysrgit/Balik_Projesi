"""
Spatial/DB fallback tests.
"""
import time

import pytest

from app.core import config
from app.db import spatial


@pytest.fixture
def fallback_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_spatial.sqlite"
    monkeypatch.setattr(config, "DB_PATH", db_path)
    monkeypatch.setattr(spatial, "_load_spatialite", lambda conn: False)
    spatial.init_db()
    return db_path


def test_init_db_creates_plain_sqlite_schema(fallback_db):
    with spatial.db_session(db_path=fallback_db) as conn:
        columns = spatial.table_columns(conn, "spatial_log")
    assert {"species", "confidence", "timestamp", "latitude", "longitude"}.issubset(columns)
    assert "geom" not in columns


def test_insert_and_query_single_detection(fallback_db):
    ts = time.time()
    spatial.insert_detection("Pufferfish", 0.85, 36.88, 30.70, ts, db_path=fallback_db)

    rows = spatial.query_detections(db_path=fallback_db)
    assert len(rows) == 1
    assert rows[0]["species"] == "Pufferfish"
    assert abs(rows[0]["latitude"] - 36.88) < 0.001
    assert abs(rows[0]["longitude"] - 30.70) < 0.001


def test_query_bbox_works_in_fallback_mode(fallback_db):
    ts = time.time()
    spatial.insert_detection("Pufferfish", 0.85, 36.88, 30.70, ts, db_path=fallback_db)
    spatial.insert_detection("Pufferfish", 0.80, 34.05, 32.45, ts + 1, db_path=fallback_db)
    spatial.insert_detection("Pufferfish", 0.75, 41.01, 29.00, ts + 2, db_path=fallback_db)

    med_rows = spatial.query_detections_bbox(33, 28, 38, 36, db_path=fallback_db)
    assert len(med_rows) == 2


def test_get_db_connection_fallback_warning(monkeypatch, tmp_path, capsys):
    db_path = tmp_path / "warn.sqlite"
    monkeypatch.setattr(spatial, "_load_spatialite", lambda conn: False)
    conn = spatial.get_db_connection(db_path=db_path)
    conn.close()

    captured = capsys.readouterr()
    assert "Falling back to plain SQLite coordinates" in captured.out
