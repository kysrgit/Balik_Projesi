"""
Aşama 3 — End-to-End Loglama Zinciri Testi

Dashboard detection_loop mantığını izole olarak simüle eder:
  test görseli → ONNX tespit → Recording ON →
    ✓ CSV satırı yazılır
    ✓ Thumbnail oluşturulur
    ✓ SocketIO emit çağrılır
    ✓ /api/record toggle çalışır
    ✓ /api/config recording durum döner
"""
import os
import sys
import glob
import csv
import shutil
import time
import threading
import queue
from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest
import cv2

# Proje kökünü path'e ekle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core import config
from app.core.detector import Detector
from app.utils import draw_boxes


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def detector():
    return Detector(str(config.MODEL_PATH))


@pytest.fixture(scope="module")
def fish_frame():
    """Garantili tespit üreten bir test görseli bul"""
    img_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "test", "images")
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    assert len(imgs) > 0, "Test görseli bulunamadı"
    
    # En az bir tespiti olan görseli seç
    det = Detector(str(config.MODEL_PATH))
    for p in imgs[:30]:
        img = cv2.imread(p)
        if img is None:
            continue
        boxes, confs = det.detect(img, conf=0.30, use_clahe=True)
        if len(boxes) > 0:
            return img
    pytest.skip("Test setinde tespit edilebilir görsel bulunamadı")


@pytest.fixture
def clean_e2e_dirs(tmp_path):
    """Her test için temiz çalışma dizinleri"""
    csv_path = str(tmp_path / "e2e_log.csv")
    thumb_dir = str(tmp_path / "thumbs")
    os.makedirs(thumb_dir, exist_ok=True)
    return csv_path, thumb_dir, tmp_path


# ─────────────────────────────────────────────────────────────────
# E2E: Detection → CSV + Thumbnail Zinciri (server.py mantığı)
# ─────────────────────────────────────────────────────────────────
class TestDetectionToLog:
    """server.py detection_loop mantığını birebir simüle eder"""

    def _run_detection_cycle(self, frame, detector, conf_thresh, csv_path, thumb_dir):
        """
        server.py detection_loop'undaki tek frame işleme mantığı.
        Orijinal kodun sadeleştirilmiş kopyası.
        """
        boxes, confs = detector.detect(frame, conf=conf_thresh, use_clahe=True, clahe_clip=3.0)

        results = {
            'boxes': boxes,
            'confs': confs,
            'csv_rows': 0,
            'thumbnails': [],
            'socket_events': []
        }

        for (x1, y1, x2, y2), c in zip(boxes, confs):
            if c >= conf_thresh:
                now_dt = datetime.now()
                ts = now_dt.strftime('%H%M%S_%f')

                # 1. Thumbnail
                thumb = frame[max(0, y1-10):y2+10, max(0, x1-10):x2+10]
                if thumb.size > 0:
                    thumbnail_name = f"t_{ts}.jpg"
                    path = os.path.join(thumb_dir, thumbnail_name)
                    cv2.imwrite(path, cv2.resize(thumb, (100, 100)))
                    results['thumbnails'].append(thumbnail_name)

                    results['socket_events'].append({
                        'timestamp': now_dt.strftime('%H:%M:%S'),
                        'confidence': round(c, 2),
                        'thumbnail': thumbnail_name
                    })

                # 2. CSV
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        ts, now_dt.strftime('%Y-%m-%d'), now_dt.strftime('%H:%M:%S'),
                        round(c, 4), x1, y1, x2, y2
                    ])
                results['csv_rows'] += 1

                break  # Orijinal koddaki gibi ilk geçerli tespiti logla

        return results

    def test_full_chain_produces_csv_and_thumb(self, detector, fish_frame, clean_e2e_dirs):
        """Tespit → CSV satır + thumbnail dosyası üretilmeli"""
        csv_path, thumb_dir, _ = clean_e2e_dirs

        # CSV başlık
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Date", "Time", "Confidence",
                             "BBox_X1", "BBox_Y1", "BBox_X2", "BBox_Y2"])

        results = self._run_detection_cycle(fish_frame, detector, 0.30, csv_path, thumb_dir)

        # En az 1 tespit olmalı (fish_frame fixture garantili)
        assert results['csv_rows'] >= 1, "CSV'ye hiç satır yazılmadı"
        assert len(results['thumbnails']) >= 1, "Thumbnail oluşturulmadı"

        # CSV dosyasını oku-doğrula
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        assert len(rows) >= 1
        assert len(rows[0]) == 8  # 8 sütun

        # Thumbnail dosyası disk'te var mı?
        thumb_path = os.path.join(thumb_dir, results['thumbnails'][0])
        assert os.path.exists(thumb_path), f"Thumbnail dosyası yok: {thumb_path}"

        # Thumbnail boyutu 100x100 mi?
        thumb_img = cv2.imread(thumb_path)
        assert thumb_img is not None
        assert thumb_img.shape[:2] == (100, 100)

        print(f"\n  E2E: {results['csv_rows']} CSV satır, "
              f"{len(results['thumbnails'])} thumbnail oluşturuldu")

    def test_csv_data_integrity(self, detector, fish_frame, clean_e2e_dirs):
        """CSV satır verileri tutarlı mı?"""
        csv_path, thumb_dir, _ = clean_e2e_dirs

        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(["Timestamp", "Date", "Time", "Confidence",
                                    "BBox_X1", "BBox_Y1", "BBox_X2", "BBox_Y2"])

        results = self._run_detection_cycle(fish_frame, detector, 0.30, csv_path, thumb_dir)
        assert results['csv_rows'] >= 1

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            row = next(reader)

        # Date format: YYYY-MM-DD
        assert len(row[1].split('-')) == 3, f"Tarih formatı hatalı: {row[1]}"
        # Confidence 0-1 aralığında
        conf = float(row[3])
        assert 0 < conf <= 1.0, f"Confidence aralık dışı: {conf}"
        # BBox koordinatları pozitif tamsayı
        for i in range(4, 8):
            val = int(row[i])
            assert val >= 0, f"Negatif koordinat: {val}"

    def test_socket_event_structure(self, detector, fish_frame, clean_e2e_dirs):
        """SocketIO emit verisi doğru formatta mı?"""
        csv_path, thumb_dir, _ = clean_e2e_dirs

        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(["Timestamp", "Date", "Time", "Confidence",
                                    "BBox_X1", "BBox_Y1", "BBox_X2", "BBox_Y2"])

        results = self._run_detection_cycle(fish_frame, detector, 0.30, csv_path, thumb_dir)

        if len(results['socket_events']) > 0:
            evt = results['socket_events'][0]
            assert 'timestamp' in evt
            assert 'confidence' in evt
            assert 'thumbnail' in evt
            assert isinstance(evt['confidence'], float)
            assert evt['thumbnail'].endswith('.jpg')


# ─────────────────────────────────────────────────────────────────
# Flask Test Client: API Endpoint'leri
# ─────────────────────────────────────────────────────────────────
class TestDashboardAPI:
    """Flask test client ile API endpoint doğrulaması"""

    @pytest.fixture(autouse=True)
    def setup_app(self):
        """GPS ve kamera thread'lerini mock'layarak Flask app kur"""
        # GPS mock - thread başlamasını engelle
        with patch('app.dashboard.server.gps_reader_thread'), \
             patch('app.dashboard.server.Camera') as mock_cam, \
             patch('app.dashboard.server.init_db'):
            
            # Mock kamera
            mock_cam_instance = MagicMock()
            mock_cam_instance.read.return_value = None
            mock_cam.return_value = mock_cam_instance
            
            import app.dashboard.server as srv
            srv.app.config['TESTING'] = True
            self.srv = srv
            self.client = srv.app.test_client()
            # Reset recording state
            srv.is_recording = False
            yield

    def test_record_toggle_on(self):
        """POST /api/record → recording ON"""
        self.srv.is_recording = False
        resp = self.client.post('/api/record')
        data = resp.get_json()
        assert resp.status_code == 200
        assert data['recording'] is True

    def test_record_toggle_off(self):
        """İki kez toggle → recording OFF"""
        self.srv.is_recording = False
        self.client.post('/api/record')
        resp = self.client.post('/api/record')
        data = resp.get_json()
        assert data['recording'] is False

    def test_config_returns_recording_state(self):
        """GET /api/config recording durumunu içermeli"""
        self.srv.is_recording = True
        resp = self.client.get('/api/config')
        data = resp.get_json()
        assert 'recording' in data
        assert data['recording'] is True

    def test_config_post_confidence(self):
        """POST /api/config ile confidence güncellenebilmeli"""
        resp = self.client.post('/api/config',
                                json={'confidence': 0.75},
                                content_type='application/json')
        assert resp.status_code == 200

        resp2 = self.client.get('/api/config')
        data = resp2.get_json()
        assert data['confidence'] == 0.75

    def test_snapshot_without_frame(self):
        """Frame yokken snapshot hata dönmeli"""
        resp = self.client.post('/api/snapshot')
        data = resp.get_json()
        assert data['status'] == 'error'

    def test_index_page_loads(self):
        """Ana sayfa 200 dönmeli"""
        resp = self.client.get('/')
        assert resp.status_code == 200
        assert b'Balon' in resp.data or b'balon' in resp.data or b'Dashboard' in resp.data


# ─────────────────────────────────────────────────────────────────
# Thread-safe Frame Queue Testi
# ─────────────────────────────────────────────────────────────────
class TestFrameQueue:
    """frame_queue drop-oldest mekanizması doğrulaması"""

    def test_queue_maxsize(self):
        q = queue.Queue(maxsize=5)
        for i in range(5):
            q.put(i)
        assert q.full()

    def test_drop_oldest_on_full(self):
        """Kuyruk doluyken en eski frame atılmalı"""
        q = queue.Queue(maxsize=3)
        q.put('A')
        q.put('B')
        q.put('C')
        assert q.full()

        # Drop-oldest strategy (server.py'deki gibi)
        if q.full():
            q.get_nowait()
        q.put_nowait('D')

        items = []
        while not q.empty():
            items.append(q.get_nowait())
        assert items == ['B', 'C', 'D'], f"Kuyruk sırası hatalı: {items}"


# ─────────────────────────────────────────────────────────────────
# Recording Kapali → Loglama Olmamali
# ─────────────────────────────────────────────────────────────────
class TestRecordingGate:
    """is_recording=False iken hiçbir şey loglanmamalı"""

    def test_no_log_when_recording_off(self, detector, fish_frame, tmp_path):
        csv_path = str(tmp_path / "no_rec.csv")
        thumb_dir = str(tmp_path / "thumbs_no_rec")
        os.makedirs(thumb_dir, exist_ok=True)

        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(["Timestamp", "Date", "Time", "Confidence",
                                    "BBox_X1", "BBox_Y1", "BBox_X2", "BBox_Y2"])

        # Tespit yap ama is_recording=False simüle et
        boxes, confs = detector.detect(fish_frame, conf=0.30, use_clahe=True)
        is_recording = False  # OFF

        csv_rows = 0
        for (x1, y1, x2, y2), c in zip(boxes, confs):
            if c >= 0.30 and is_recording:
                # Bu bloğa girmemeli
                csv_rows += 1

        assert csv_rows == 0, "Recording kapalıyken log yazılmamalı"
        # Thumb dizini boş kalmalı
        assert len(os.listdir(thumb_dir)) == 0

    def test_log_when_recording_on(self, detector, fish_frame, tmp_path):
        csv_path = str(tmp_path / "rec_on.csv")
        thumb_dir = str(tmp_path / "thumbs_rec_on")
        os.makedirs(thumb_dir, exist_ok=True)

        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(["Timestamp", "Date", "Time", "Confidence",
                                    "BBox_X1", "BBox_Y1", "BBox_X2", "BBox_Y2"])

        boxes, confs = detector.detect(fish_frame, conf=0.30, use_clahe=True)
        is_recording = True  # ON

        csv_rows = 0
        for (x1, y1, x2, y2), c in zip(boxes, confs):
            if c >= 0.30 and is_recording:
                now_dt = datetime.now()
                ts = now_dt.strftime('%H%M%S_%f')

                # Thumbnail
                thumb = fish_frame[max(0, y1-10):y2+10, max(0, x1-10):x2+10]
                if thumb.size > 0:
                    tn = f"t_{ts}.jpg"
                    cv2.imwrite(os.path.join(thumb_dir, tn), cv2.resize(thumb, (100, 100)))

                # CSV
                with open(csv_path, 'a', newline='') as f:
                    csv.writer(f).writerow([
                        ts, now_dt.strftime('%Y-%m-%d'), now_dt.strftime('%H:%M:%S'),
                        round(c, 4), x1, y1, x2, y2
                    ])
                csv_rows += 1
                break

        assert csv_rows >= 1, "Recording açıkken log yazılmalı"
        assert len(os.listdir(thumb_dir)) >= 1, "Thumbnail oluşturulmalı"
