"""
Aşama 4 — Veri Paylaşım Katmanı Testleri

Export formatları (GeoJSON, CSV, DarwinCore) ve webhook sistemi doğrulaması.
"""
import os
import sys
import csv
import json
import zipfile
import io
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.export.formats import to_geojson, to_csv_download, to_darwincore_archive, _read_detections_csv
from app.export.webhook import WebhookNotifier


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_csv(tmp_path):
    """Örnek tespit CSV dosyası"""
    csv_path = str(tmp_path / "test_detections.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Date", "Time", "Confidence",
                         "BBox_X1", "BBox_Y1", "BBox_X2", "BBox_Y2"])
        writer.writerow(["120000_000000", "2026-03-04", "12:00:00", "0.9234", "10", "20", "100", "200"])
        writer.writerow(["120001_000000", "2026-03-04", "12:00:01", "0.8567", "50", "60", "150", "250"])
        writer.writerow(["120002_000000", "2026-03-04", "12:00:02", "0.7100", "30", "40", "120", "180"])
    return csv_path


@pytest.fixture
def sample_db(tmp_path):
    """Örnek spatial veritabanı (düz SQLite)"""
    import sqlite3
    db_path = str(tmp_path / "test_spatial.sqlite")
    conn = sqlite3.connect(db_path)
    conn.execute('''CREATE TABLE spatial_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        species TEXT, confidence REAL, timestamp REAL,
        latitude REAL, longitude REAL
    )''')
    ts = time.time()
    conn.execute("INSERT INTO spatial_log VALUES (NULL, 'Pufferfish', 0.92, ?, 36.88, 30.70)", (ts,))
    conn.execute("INSERT INTO spatial_log VALUES (NULL, 'Pufferfish', 0.85, ?, 34.05, 32.45)", (ts+1,))
    conn.execute("INSERT INTO spatial_log VALUES (NULL, 'Pufferfish', 0.78, ?, 36.10, 29.05)", (ts+2,))
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def empty_csv(tmp_path):
    """Boş CSV (sadece başlık)"""
    csv_path = str(tmp_path / "empty.csv")
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(["Timestamp", "Date", "Time", "Confidence",
                                "BBox_X1", "BBox_Y1", "BBox_X2", "BBox_Y2"])
    return csv_path


# ─────────────────────────────────────────────────────────────────
# 1. GeoJSON Export
# ─────────────────────────────────────────────────────────────────
class TestGeoJSON:
    def test_geojson_structure(self, sample_csv):
        result = to_geojson(sample_csv)
        assert result["type"] == "FeatureCollection"
        assert "features" in result
        assert len(result["features"]) == 3

    def test_geojson_feature_properties(self, sample_csv):
        result = to_geojson(sample_csv)
        props = result["features"][0]["properties"]
        assert "species" in props
        assert "confidence" in props
        assert props["source"] == "antigravity_pufferfish_detector"

    def test_geojson_with_db_gps(self, sample_csv, sample_db):
        """DB varsa GPS koordinatları GeoJSON'a eklenmeli"""
        result = to_geojson(sample_csv, sample_db)
        assert len(result["features"]) == 3
        feat = result["features"][0]
        assert feat["geometry"]["type"] == "Point"
        coords = feat["geometry"]["coordinates"]
        assert len(coords) == 2
        assert abs(coords[1] - 36.88) < 0.01  # lat
        assert abs(coords[0] - 30.70) < 0.01  # lon (GeoJSON: [lon, lat])

    def test_geojson_empty_csv(self, empty_csv):
        result = to_geojson(empty_csv)
        assert result["type"] == "FeatureCollection"
        assert len(result["features"]) == 0

    def test_geojson_serializable(self, sample_csv):
        result = to_geojson(sample_csv)
        json_str = json.dumps(result, ensure_ascii=False)
        assert len(json_str) > 0
        parsed = json.loads(json_str)
        assert parsed["type"] == "FeatureCollection"

    def test_geojson_crs(self, sample_csv):
        result = to_geojson(sample_csv)
        assert "crs" in result
        assert "CRS84" in result["crs"]["properties"]["name"]


# ─────────────────────────────────────────────────────────────────
# 2. CSV Download
# ─────────────────────────────────────────────────────────────────
class TestCSVDownload:
    def test_csv_output_has_header(self, sample_csv):
        output = to_csv_download(sample_csv)
        lines = output.strip().split('\n')
        assert len(lines) == 4  # header + 3 rows

    def test_csv_with_db_uses_gps_columns(self, sample_csv, sample_db):
        output = to_csv_download(sample_csv, sample_db)
        lines = output.strip().split('\n')
        header = lines[0]
        assert "Latitude" in header
        assert "Longitude" in header
        assert len(lines) == 4  # header + 3 DB rows

    def test_csv_empty(self, empty_csv):
        output = to_csv_download(empty_csv)
        lines = output.strip().split('\n')
        assert len(lines) == 1  # Only header

    def test_csv_parseable(self, sample_csv):
        output = to_csv_download(sample_csv)
        reader = csv.reader(io.StringIO(output))
        rows = list(reader)
        assert len(rows) == 4
        assert len(rows[0]) == 8  # 8 column header


# ─────────────────────────────────────────────────────────────────
# 3. DarwinCore Archive
# ─────────────────────────────────────────────────────────────────
class TestDarwinCore:
    def test_zip_structure(self, sample_csv):
        zip_data = to_darwincore_archive(sample_csv)
        assert len(zip_data) > 0

        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            names = zf.namelist()
            assert "occurrence.csv" in names
            assert "meta.xml" in names
            assert "eml.xml" in names

    def test_occurrence_csv_content(self, sample_csv):
        zip_data = to_darwincore_archive(sample_csv)
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            occ = zf.read("occurrence.csv").decode('utf-8')
            lines = occ.strip().split('\n')
            assert len(lines) == 4  # header + 3 occurrences

            # DarwinCore sütun sayısı kontrolü
            header_cols = lines[0].split('\t')
            assert "occurrenceID" in header_cols[0]
            assert "basisOfRecord" in header_cols[1]
            assert "scientificName" in header_cols[3]

    def test_occurrence_darwincore_fields(self, sample_csv):
        zip_data = to_darwincore_archive(sample_csv)
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            occ = zf.read("occurrence.csv").decode('utf-8')
            reader = csv.reader(io.StringIO(occ), delimiter='\t')
            header = next(reader)
            row = next(reader)

            # Zorunlu DarwinCore alanları
            assert row[0].startswith("AG-PF-")  # occurrenceID
            assert row[1] == "MachineObservation"  # basisOfRecord
            assert row[3] == "Lagocephalus sceleratus"  # scientificName
            assert row[5] == "Animalia"  # kingdom
            assert row[9] == "Tetraodontidae"  # family

    def test_meta_xml_valid(self, sample_csv):
        zip_data = to_darwincore_archive(sample_csv)
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            meta = zf.read("meta.xml").decode('utf-8')
            assert '<?xml' in meta
            assert 'dwc/terms/Occurrence' in meta
            assert 'occurrence.csv' in meta

    def test_eml_xml_metadata(self, sample_csv):
        zip_data = to_darwincore_archive(sample_csv)
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            eml = zf.read("eml.xml").decode('utf-8')
            assert 'Lagocephalus sceleratus' in eml
            assert 'Mediterranean' in eml
            assert 'Antigravity' in eml
            assert 'CC BY 4.0' in eml

    def test_darwincore_with_db(self, sample_csv, sample_db):
        """DB verisi GPS koordinatlarını DarwinCore'a eklemeli"""
        zip_data = to_darwincore_archive(sample_csv, sample_db)
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            occ = zf.read("occurrence.csv").decode('utf-8')
            reader = csv.reader(io.StringIO(occ), delimiter='\t')
            header = next(reader)
            row = next(reader)
            
            lat_idx = header.index("decimalLatitude")
            lon_idx = header.index("decimalLongitude")
            assert float(row[lat_idx]) > 0  # Akdeniz kuzey yarıküre
            assert float(row[lon_idx]) > 0


# ─────────────────────────────────────────────────────────────────
# 4. Webhook Notifier
# ─────────────────────────────────────────────────────────────────
class TestWebhookNotifier:
    def test_add_remove_target(self):
        n = WebhookNotifier()
        n.add_target("slack", "https://hooks.slack.com/test")
        assert "slack" in n.get_targets()
        n.remove_target("slack")
        assert "slack" not in n.get_targets()

    def test_get_targets_copy(self):
        """Dışarıdan modifikasyona karşı güvenli"""
        n = WebhookNotifier()
        n.add_target("test", "https://example.com")
        targets = n.get_targets()
        targets["hacker"] = "evil"
        assert "hacker" not in n.get_targets()

    def test_rate_limiting(self):
        n = WebhookNotifier(rate_limit_seconds=10)
        n.add_target("test", "https://httpbin.org/post")  # Won't actually send
        
        # İlk çağrı (rate limit yok)
        with patch.object(n, '_send_one', return_value=True):
            n.notify(species="test", confidence=0.8)
        
        # İkinci çağrı hemen sonra (rate limited)
        result = n.notify(species="test", confidence=0.8)
        assert result == {}  # Rate limited
        assert n.get_stats()['rate_limited'] >= 1

    def test_empty_targets_no_error(self):
        n = WebhookNotifier(rate_limit_seconds=0)
        result = n.notify(species="test", confidence=0.8)
        assert result == {}

    def test_format_payload_generic(self):
        n = WebhookNotifier()
        n.add_target("custom", "https://myapi.example.com/hook")
        payload = n._format_payload("custom", {
            'species': 'Lagocephalus sceleratus',
            'confidence': 0.85,
            'lat': 36.88,
            'lon': 30.70,
            'timestamp': '2026-03-04 12:00:00'
        })
        parsed = json.loads(payload)
        assert parsed["event"] == "pufferfish_detection"
        assert parsed["data"]["confidence"] == 0.85

    def test_format_payload_slack(self):
        n = WebhookNotifier()
        n.add_target("slack", "https://hooks.slack.com/test")
        payload = n._format_payload("slack", {
            'species': 'Lagocephalus sceleratus',
            'confidence': 0.9,
            'lat': None, 'lon': None,
            'timestamp': '12:00:00'
        })
        parsed = json.loads(payload)
        assert "text" in parsed
        assert "Balon Balığı" in parsed["text"]

    def test_format_payload_discord(self):
        n = WebhookNotifier()
        n.add_target("discord", "https://discord.com/api/webhooks/test")
        payload = n._format_payload("discord", {
            'species': 'test', 'confidence': 0.7,
            'lat': None, 'lon': None, 'timestamp': ''
        })
        parsed = json.loads(payload)
        assert "content" in parsed

    def test_stats_tracking(self):
        n = WebhookNotifier()
        stats = n.get_stats()
        assert stats['sent'] == 0
        assert stats['failed'] == 0
        assert stats['rate_limited'] == 0


# ─────────────────────────────────────────────────────────────────
# 5. Flask Export Endpoints
# ─────────────────────────────────────────────────────────────────
class TestExportEndpoints:
    """Flask test client ile export endpoint doğrulaması"""

    @pytest.fixture(autouse=True)
    def setup_app(self):
        with patch('app.dashboard.server.gps_reader_thread'), \
             patch('app.dashboard.server.Camera') as mock_cam, \
             patch('app.dashboard.server.init_db'):
            mock_cam_instance = MagicMock()
            mock_cam_instance.read.return_value = None
            mock_cam.return_value = mock_cam_instance

            import app.dashboard.server as srv
            srv.app.config['TESTING'] = True
            self.client = srv.app.test_client()
            self.srv = srv
            yield

    def test_geojson_endpoint(self):
        resp = self.client.get('/api/export/geojson')
        assert resp.status_code == 200
        assert 'geo+json' in resp.content_type
        data = json.loads(resp.data)
        assert data["type"] == "FeatureCollection"

    def test_csv_endpoint(self):
        resp = self.client.get('/api/export/csv')
        assert resp.status_code == 200
        assert 'text/csv' in resp.content_type

    def test_darwincore_endpoint(self):
        resp = self.client.get('/api/export/darwincore')
        assert resp.status_code == 200
        assert 'application/zip' in resp.content_type
        # ZIP olarak açılabilir mi?
        zf = zipfile.ZipFile(io.BytesIO(resp.data))
        assert "occurrence.csv" in zf.namelist()
        zf.close()

    def test_webhook_crud(self):
        # POST: Ekle
        resp = self.client.post('/api/webhooks',
                                json={'name': 'test', 'url': 'https://example.com/hook'},
                                content_type='application/json')
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'test' in data['targets']

        # GET: Listele
        resp = self.client.get('/api/webhooks')
        data = resp.get_json()
        assert 'test' in data['targets']
        assert 'stats' in data

        # DELETE: Sil
        resp = self.client.delete('/api/webhooks',
                                  json={'name': 'test'},
                                  content_type='application/json')
        data = resp.get_json()
        assert 'test' not in data['targets']

    def test_webhook_validation(self):
        resp = self.client.post('/api/webhooks',
                                json={'name': '', 'url': ''},
                                content_type='application/json')
        assert resp.status_code == 400
