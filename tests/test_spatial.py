"""
Aşama 2 — Spatial/DB Testi
SpatiaLite extension olmadan düz SQLite ile veritabanı
insert/select işlemlerini doğrular.
"""
import os
import sqlite3
import time
import tempfile
from unittest.mock import patch, MagicMock
import pytest

from app.db.spatial import get_db_connection

class TestSpatialFallback:
    """SpatiaLite olmayan ortamda (Windows/CI) düz SQLite ile test"""

    @pytest.fixture(autouse=True)
    def setup_db(self, tmp_path):
        """Her test için geçici veritabanı oluştur"""
        self.db_path = str(tmp_path / "test_spatial.sqlite")
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        
        # SpatiaLite olmadan sadece düz tablo oluştur
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS spatial_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                species TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp REAL NOT NULL,
                latitude REAL,
                longitude REAL
            )
        ''')
        self.conn.commit()
        yield
        self.conn.close()

    def _insert(self, species, confidence, lat, lon, ts):
        """SpatiaLite MakePoint yerine düz lat/lon insert"""
        self.conn.execute(
            "INSERT INTO spatial_log (species, confidence, timestamp, latitude, longitude) VALUES (?, ?, ?, ?, ?)",
            (species, confidence, ts, lat, lon)
        )
        self.conn.commit()

    def test_insert_single(self):
        ts = time.time()
        self._insert("Pufferfish", 0.85, 36.88, 30.70, ts)
        
        row = self.conn.execute("SELECT * FROM spatial_log").fetchone()
        assert row is not None
        assert row["species"] == "Pufferfish"
        assert row["confidence"] == 0.85
        assert abs(row["latitude"] - 36.88) < 0.001
        assert abs(row["longitude"] - 30.70) < 0.001

    def test_insert_multiple(self):
        ts = time.time()
        self._insert("Pufferfish", 0.90, 36.00, 30.00, ts)
        self._insert("Pufferfish", 0.75, 37.00, 31.00, ts + 1)
        self._insert("Pufferfish", 0.60, 38.00, 32.00, ts + 2)
        
        rows = self.conn.execute("SELECT * FROM spatial_log ORDER BY id").fetchall()
        assert len(rows) == 3
        assert rows[0]["confidence"] == 0.90
        assert rows[2]["latitude"] == 38.00

    def test_query_by_confidence(self):
        ts = time.time()
        self._insert("Pufferfish", 0.95, 36.88, 30.70, ts)
        self._insert("Pufferfish", 0.40, 36.88, 30.70, ts + 1)
        self._insert("Pufferfish", 0.80, 36.88, 30.70, ts + 2)
        
        high_conf = self.conn.execute(
            "SELECT * FROM spatial_log WHERE confidence >= 0.70"
        ).fetchall()
        assert len(high_conf) == 2

    def test_query_by_region(self):
        """Basit bounding-box sorgusu (SpatiaLite spatial index yerine)"""
        ts = time.time()
        # Akdeniz noktaları
        self._insert("Pufferfish", 0.85, 36.88, 30.70, ts)      # Antalya
        self._insert("Pufferfish", 0.80, 34.05, 32.45, ts + 1)  # Kıbrıs
        self._insert("Pufferfish", 0.75, 41.01, 29.00, ts + 2)  # İstanbul (Akdeniz dışı)
        
        # Akdeniz kuşağı: lat 33-38, lon 28-36
        med_rows = self.conn.execute(
            "SELECT * FROM spatial_log WHERE latitude BETWEEN 33 AND 38 AND longitude BETWEEN 28 AND 36"
        ).fetchall()
        assert len(med_rows) == 2  # Antalya + Kıbrıs

    def test_table_schema(self):
        """Tablo sütunları doğru mu?"""
        cursor = self.conn.execute("PRAGMA table_info(spatial_log)")
        columns = [row[1] for row in cursor.fetchall()]
        assert "species" in columns
        assert "confidence" in columns
        assert "timestamp" in columns
        assert "latitude" in columns
        assert "longitude" in columns

    def test_empty_table(self):
        rows = self.conn.execute("SELECT COUNT(*) FROM spatial_log").fetchone()
        assert rows[0] == 0

    @patch('app.db.spatial.sqlite3.connect')
    def test_get_db_connection_fallback_warning(self, mock_connect, capsys):
        """Test the untested error path in get_db_connection"""
        # Create a mock connection
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        # The execute method should always raise sqlite3.OperationalError
        mock_conn.execute.side_effect = sqlite3.OperationalError("Could not load extension")

        # Call the method
        conn = get_db_connection()

        # Check that we got the mocked connection back
        assert conn == mock_conn

        # Assert the standard warning was printed
        captured = capsys.readouterr()
        assert "WARNING: Could not load SpatiaLite extension. Spatial indexing will not work." in captured.out

        # Verify execute was called 4 times with the different extension attempts
        assert mock_conn.execute.call_count == 4
        calls = mock_conn.execute.call_args_list
        assert calls[0][0][0] == "SELECT load_extension('mod_spatialite')"
        assert calls[1][0][0] == "SELECT load_extension('libspatialite.so')"
        assert calls[2][0][0] == "SELECT load_extension('mod_spatialite.dylib')"
        assert calls[3][0][0] == "SELECT load_extension('mod_spatialite.dll')"
