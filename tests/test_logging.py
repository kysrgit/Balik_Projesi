"""
Aşama 2 — CSV Loglama Testi
CSV dosyasına yazma/okuma ve format doğrulaması.
"""
import os
import csv
import tempfile
import pytest

from app.main import _ensure_csv_header, _log_detection_csv, CSV_LOG_FILE


class TestCSVHeader:
    def test_header_created_when_missing(self, tmp_path):
        """Dosya yokken başlık satırı oluşturulmalı"""
        import app.main as main_mod
        original = main_mod.CSV_LOG_FILE
        test_csv = str(tmp_path / "test_log.csv")
        main_mod.CSV_LOG_FILE = test_csv

        try:
            _ensure_csv_header()
            assert os.path.exists(test_csv)

            with open(test_csv, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
            
            expected = ["Timestamp", "Date", "Time", "Confidence", "BBox_X1", "BBox_Y1", "BBox_X2", "BBox_Y2"]
            assert header == expected, f"Başlık uyumsuz: {header}"
        finally:
            main_mod.CSV_LOG_FILE = original

    def test_header_not_duplicated(self, tmp_path):
        """Dosya zaten varsa ikinci başlık eklenmemeli"""
        import app.main as main_mod
        original = main_mod.CSV_LOG_FILE
        test_csv = str(tmp_path / "test_log2.csv")
        main_mod.CSV_LOG_FILE = test_csv

        try:
            _ensure_csv_header()
            _ensure_csv_header()  # İkinci çağrı

            with open(test_csv, 'r') as f:
                lines = f.readlines()
            
            # Sadece 1 satır (başlık) olmalı
            assert len(lines) == 1, f"Başlık çoğaltılmış: {len(lines)} satır"
        finally:
            main_mod.CSV_LOG_FILE = original


class TestCSVWriting:
    def test_detection_logged(self, tmp_path):
        """Tespit verisi CSV'ye doğru formatta yazılmalı"""
        import app.main as main_mod
        original = main_mod.CSV_LOG_FILE
        test_csv = str(tmp_path / "test_det.csv")
        main_mod.CSV_LOG_FILE = test_csv

        try:
            _ensure_csv_header()
            
            boxes = [(10, 20, 100, 200), (50, 60, 150, 250)]
            confs = [0.85, 0.72]
            _log_detection_csv(boxes, confs)

            with open(test_csv, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                rows = list(reader)

            assert len(rows) == 2, f"2 tespit beklendi, {len(rows)} bulundu"
            
            # İlk satırın confidence değerini kontrol et
            assert float(rows[0][3]) == 0.85
            assert int(rows[0][4]) == 10  # BBox_X1
            assert int(rows[0][5]) == 20  # BBox_Y1
            
            # İkinci satır
            assert float(rows[1][3]) == 0.72
        finally:
            main_mod.CSV_LOG_FILE = original

    def test_empty_detection_no_rows(self, tmp_path):
        """Boş tespit listesi satır eklememeli"""
        import app.main as main_mod
        original = main_mod.CSV_LOG_FILE
        test_csv = str(tmp_path / "test_empty.csv")
        main_mod.CSV_LOG_FILE = test_csv

        try:
            _ensure_csv_header()
            _log_detection_csv([], [])

            with open(test_csv, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                rows = list(reader)

            assert len(rows) == 0
        finally:
            main_mod.CSV_LOG_FILE = original

    def test_multiple_writes_append(self, tmp_path):
        """Ardışık yazımlar üst üste eklenmeli"""
        import app.main as main_mod
        original = main_mod.CSV_LOG_FILE
        test_csv = str(tmp_path / "test_append.csv")
        main_mod.CSV_LOG_FILE = test_csv

        try:
            _ensure_csv_header()
            _log_detection_csv([(1, 2, 3, 4)], [0.9])
            _log_detection_csv([(5, 6, 7, 8)], [0.8])

            with open(test_csv, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                rows = list(reader)

            assert len(rows) == 2
        finally:
            main_mod.CSV_LOG_FILE = original
