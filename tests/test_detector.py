"""
Aşama 2 — Offline Inference Testi
dataset/test/images/ klasöründeki gerçek balon balığı fotoğrafları üzerinde
ONNX modelinin tespit yapabildiğini doğrular.
"""
import os
import glob
import pytest
import cv2
import numpy as np

from app.core import config
from app.core.detector import Detector
from app.utils.image import apply_clahe, draw_boxes


# ------------------------------------------------------------------
# Fixture: Modeli bir kez yükle, tüm testlerde paylaş
# ------------------------------------------------------------------
@pytest.fixture(scope="module")
def detector():
    model_path = config.MODEL_PATH
    assert os.path.exists(model_path), f"Model dosyasi bulunamadi: {model_path}"
    return Detector(str(model_path))


@pytest.fixture(scope="module")
def test_images():
    """dataset/test/images/ klasöründen jpg dosyalarını topla"""
    img_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "test", "images")
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    assert len(imgs) > 0, f"Test görseli bulunamadı: {img_dir}"
    return imgs


# ------------------------------------------------------------------
# 1. Model yükleme testi
# ------------------------------------------------------------------
class TestModelLoading:
    def test_model_exists(self):
        assert os.path.exists(config.MODEL_PATH), "ONNX model dosyası eksik"

    def test_model_loads(self, detector):
        assert detector is not None
        assert detector.model is not None


# ------------------------------------------------------------------
# 2. Tek görsel üzerinde inference
# ------------------------------------------------------------------
class TestSingleInference:
    def test_inference_returns_tuple(self, detector, test_images):
        img = cv2.imread(test_images[0])
        assert img is not None, "Görsel okunamadı"
        boxes, confs = detector.detect(img, conf=0.3, use_clahe=True, clahe_clip=3.0)
        assert isinstance(boxes, list)
        assert isinstance(confs, list)
        assert len(boxes) == len(confs)

    def test_box_format(self, detector, test_images):
        img = cv2.imread(test_images[0])
        boxes, confs = detector.detect(img, conf=0.3, use_clahe=True)
        for box in boxes:
            assert len(box) == 4, "Kutu (x1, y1, x2, y2) formatında olmalı"
            x1, y1, x2, y2 = box
            assert x2 > x1 and y2 > y1, "Kutu koordinatları geçersiz"

    def test_confidence_range(self, detector, test_images):
        img = cv2.imread(test_images[0])
        _, confs = detector.detect(img, conf=0.2, use_clahe=True)
        for c in confs:
            assert 0.0 <= c <= 1.0, f"Confidence {c} aralık dışı"


# ------------------------------------------------------------------
# 3. Toplu inference — en az birkaç görselde tespit olmalı
# ------------------------------------------------------------------
class TestBatchInference:
    def test_at_least_some_detections(self, detector, test_images):
        """Test setindeki görsellerin en az %20'sinde tespit olmalı"""
        detected_count = 0
        sample = test_images[:20]  # İlk 20 görselle sınırla (hız)

        for img_path in sample:
            img = cv2.imread(img_path)
            if img is None:
                continue
            boxes, confs = detector.detect(img, conf=0.3, use_clahe=True)
            if len(boxes) > 0:
                detected_count += 1

        ratio = detected_count / len(sample)
        assert ratio >= 0.2, (
            f"Tespit oranı çok düşük: {detected_count}/{len(sample)} = {ratio:.0%}. "
            f"Model çalışmıyor olabilir."
        )
        print(f"\n  Toplu inference: {detected_count}/{len(sample)} görselde tespit ({ratio:.0%})")


# ------------------------------------------------------------------
# 4. CLAHE açık / kapalı karşılaştırma
# ------------------------------------------------------------------
class TestCLAHE:
    def test_clahe_produces_different_image(self, test_images):
        img = cv2.imread(test_images[0])
        enhanced = apply_clahe(img, clip=3.0)
        assert enhanced is not None
        assert enhanced.shape == img.shape
        # CLAHE sonrası görsel orijinalden farklı olmalı
        assert not np.array_equal(img, enhanced), "CLAHE hiçbir değişiklik yapmadı"

    def test_inference_works_without_clahe(self, detector, test_images):
        img = cv2.imread(test_images[0])
        boxes, confs = detector.detect(img, conf=0.3, use_clahe=False)
        assert isinstance(boxes, list)

    def test_inference_works_with_clahe(self, detector, test_images):
        img = cv2.imread(test_images[0])
        boxes, confs = detector.detect(img, conf=0.3, use_clahe=True, clahe_clip=3.0)
        assert isinstance(boxes, list)


# ------------------------------------------------------------------
# 5. draw_boxes fonksiyonu
# ------------------------------------------------------------------
class TestDrawBoxes:
    def test_draw_boxes_no_detections(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = draw_boxes(frame, [], [])
        assert result.shape == frame.shape

    def test_draw_boxes_with_detections(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = [(10, 10, 100, 100)]
        confs = [0.85]
        result = draw_boxes(frame, boxes, confs)
        assert result.shape == frame.shape
        # Kutu çizilmiş olmalı — en az bir piksel kırmızı renkte
        assert np.any(result[:, :, 2] > 0), "Kutu çizilmemiş görünüyor"
