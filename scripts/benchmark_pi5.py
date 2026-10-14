#!/usr/bin/env python3
"""Reproducible ONNX FP32 benchmark for Raspberry Pi 5 deployments.

This script benchmarks the deployed ONNX detector in two modes:
1. image_folder: sorted images read from disk
2. live_camera: Picamera2 when available, otherwise OpenCV fallback

It records detailed runtime metadata, per-frame timing, aggregate latency/FPS
metrics, and manuscript-ready summary artifacts.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

try:
    import torch
except Exception:  # pragma: no cover - optional fallback path
    torch = None

try:
    from ultralytics.utils.nms import non_max_suppression
except Exception as exc:  # pragma: no cover - optional fallback path
    non_max_suppression = None
    ULTRALYTICS_NMS_IMPORT_ERROR = str(exc)
else:
    ULTRALYTICS_NMS_IMPORT_ERROR = None


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DEFAULT_CLAHE_CLIP = 3.0
DEFAULT_CLAHE_GRID = (8, 8)
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480
DEFAULT_CAMERA_FPS = 30
NMS_IOU_THRESHOLD = 0.45
IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
THROTTLED_FLAGS = {
    0: "under_voltage_now",
    1: "arm_frequency_capped_now",
    2: "throttled_now",
    3: "soft_temperature_limit_now",
    16: "under_voltage_occurred",
    17: "arm_frequency_capped_occurred",
    18: "throttling_occurred",
    19: "soft_temperature_limit_occurred",
}

_CLAHE_CACHE: dict[tuple[float, tuple[int, int]], cv2.CLAHE] = {}


@dataclass
class FramePacket:
    frame: np.ndarray
    frame_id: int
    captured_at: float
    capture_duration_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the deployed ONNX detector for Raspberry Pi 5."
    )
    parser.add_argument("--model", required=True, help="Path to the ONNX model file.")
    parser.add_argument(
        "--image-dir",
        help="Path to an image directory. Required for --mode image_folder.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["image_folder", "live_camera"],
        help="Benchmark mode.",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=20,
        help="Number of warmup iterations excluded from reported results.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=200,
        help="Number of measured benchmark iterations.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference input size. For the bundled ONNX model this must be 640.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.60,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--clahe",
        choices=["on", "off"],
        default="on",
        help="Enable or disable CLAHE preprocessing.",
    )
    parser.add_argument(
        "--provider",
        choices=["xnnpack", "cpu"],
        default="xnnpack",
        help="Requested ONNX Runtime execution provider.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path for the JSON result artifact.",
    )
    parser.add_argument(
        "--output-csv",
        help="Optional path for the CSV result artifact.",
    )
    parser.add_argument(
        "--save-summary-md",
        help="Optional path for the Markdown summary artifact.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def now_local_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def timestamp_slug() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def bool_to_yes_no(value: bool) -> str:
    return "yes" if value else "no"


def on_off_to_bool(value: str) -> bool:
    return value.strip().lower() == "on"


def format_number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:.{digits}f}"


def read_text_file(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").strip("\x00 \n\r\t")
    except OSError:
        return None


def run_command(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = (completed.stdout or completed.stderr or "").strip()
    return output or None


def parse_os_release() -> dict[str, str]:
    data: dict[str, str] = {}
    os_release = Path("/etc/os-release")
    if not os_release.exists():
        return data
    try:
        for line in os_release.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "=" not in line or line.startswith("#"):
                continue
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip().strip('"')
    except OSError:
        return {}
    return data


def detect_raspberry_pi_model() -> str | None:
    for path_str in (
        "/sys/firmware/devicetree/base/model",
        "/proc/device-tree/model",
    ):
        value = read_text_file(Path(path_str))
        if value:
            return value
    cpuinfo = read_text_file(Path("/proc/cpuinfo"))
    if not cpuinfo:
        return None
    for line in cpuinfo.splitlines():
        if line.lower().startswith("model"):
            _, value = line.split(":", 1)
            return value.strip()
    return None


def detect_cpu_model() -> str | None:
    cpuinfo = read_text_file(Path("/proc/cpuinfo"))
    if cpuinfo:
        for key in ("Model", "model name", "Hardware", "Processor"):
            prefix = f"{key}:"
            for line in cpuinfo.splitlines():
                if line.startswith(prefix):
                    return line.split(":", 1)[1].strip()
    processor = platform.processor()
    return processor or None


def detect_ram_mb() -> int | None:
    meminfo = read_text_file(Path("/proc/meminfo"))
    if not meminfo:
        return None
    for line in meminfo.splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(round(int(parts[1]) / 1024.0))
    return None


def parse_cpu_temperature_output(raw: str) -> float | None:
    if not raw:
        return None
    text = raw.strip().lower().replace("temp=", "").replace("'c", "").replace("c", "")
    try:
        return float(text)
    except ValueError:
        return None


def read_cpu_temperature_c() -> float | None:
    thermal_path = Path("/sys/class/thermal/thermal_zone0/temp")
    if thermal_path.exists():
        raw = read_text_file(thermal_path)
        if raw:
            try:
                value = float(raw)
            except ValueError:
                value = None
            if value is not None:
                return value / 1000.0 if value > 200 else value
    if shutil.which("vcgencmd"):
        return parse_cpu_temperature_output(run_command(["vcgencmd", "measure_temp"]) or "")
    return None


def parse_throttled_status(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    raw = raw.strip()
    if "0x" not in raw:
        return {"raw": raw, "hex": None, "active_flags": []}
    hex_value = raw.split("0x", 1)[1]
    try:
        bits = int(hex_value, 16)
    except ValueError:
        return {"raw": raw, "hex": f"0x{hex_value}", "active_flags": []}
    active_flags = [name for bit, name in THROTTLED_FLAGS.items() if bits & (1 << bit)]
    return {"raw": raw, "hex": f"0x{hex_value}", "active_flags": active_flags}


def read_throttled_status() -> dict[str, Any] | None:
    if not shutil.which("vcgencmd"):
        return None
    return parse_throttled_status(run_command(["vcgencmd", "get_throttled"]))


def detect_is_raspberry_pi_5(rpi_model: str | None, cpu_model: str | None) -> bool:
    haystack = " ".join(part for part in (rpi_model, cpu_model) if part).lower()
    return "raspberry pi 5" in haystack or "bcm2712" in haystack


def apply_clahe(img: np.ndarray, clip: float = DEFAULT_CLAHE_CLIP, grid: tuple[int, int] = DEFAULT_CLAHE_GRID) -> np.ndarray:
    if img is None:
        raise ValueError("Cannot apply CLAHE to an empty image.")
    key = (float(clip), tuple(grid))
    clahe = _CLAHE_CACHE.get(key)
    if clahe is None:
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
        _CLAHE_CACHE[key] = clahe
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    enhanced_l = clahe.apply(l_channel)
    merged = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def list_image_files(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    files = sorted(
        path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"No supported image files found under: {image_dir}")
    return files


def summarize_series(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "std": float(np.std(array)),
    }


def fps_from_latency_ms(values: list[float]) -> list[float]:
    fps_values: list[float] = []
    for value in values:
        if value and value > 0:
            fps_values.append(1000.0 / value)
    return fps_values


def progress_tick(prefix: str, current_index: int, total: int) -> None:
    if total <= 0:
        return
    checkpoint = max(1, total // 10)
    is_last = current_index == total
    if current_index == 1 or current_index % checkpoint == 0 or is_last:
        print(f"[{prefix}] {current_index}/{total}")


class BenchmarkCamera:
    """Continuous camera capture with Picamera2 first and OpenCV fallback."""

    def __init__(self, width: int, height: int, fps: int = DEFAULT_CAMERA_FPS) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.backend = "uninitialized"
        self.warning: str | None = None
        self.picam = None
        self.cap = None
        self._condition = threading.Condition()
        self._running = False
        self._thread: threading.Thread | None = None
        self._latest_frame: np.ndarray | None = None
        self._latest_frame_id = 0
        self._latest_captured_at = 0.0
        self._latest_capture_duration_ms = 0.0
        self._capture_count = 0
        self._capture_started_at = 0.0
        self._capture_last_at = 0.0
        self._capture_failures = 0
        self._initialize_backend()

    def _initialize_backend(self) -> None:
        if self._init_picamera2():
            return
        if self._init_opencv():
            return
        raise RuntimeError(
            "Could not initialize Picamera2 or OpenCV camera backend. "
            f"Last warning: {self.warning or 'unknown'}"
        )

    def _init_picamera2(self) -> bool:
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            self.warning = f"Picamera2 import failed: {exc}"
            return False
        try:
            picam = Picamera2()
            config = picam.create_video_configuration(main={"size": (self.width, self.height)})
            picam.configure(config)
            picam.start()
        except Exception as exc:  # pragma: no cover - hardware-specific
            self.warning = f"Picamera2 initialization failed: {exc}"
            return False
        self.picam = picam
        self.backend = "picamera2"
        return True

    def _init_opencv(self) -> bool:
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.warning = "OpenCV VideoCapture(0) could not be opened."
                return False
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
        except Exception as exc:  # pragma: no cover - hardware-specific
            self.warning = f"OpenCV camera initialization failed: {exc}"
            return False
        self.cap = cap
        self.backend = "opencv_videocapture"
        return True

    def start(self) -> "BenchmarkCamera":
        self._running = True
        self._capture_started_at = time.perf_counter()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return self

    def _capture_loop(self) -> None:  # pragma: no cover - hardware-specific
        while self._running:
            start = time.perf_counter()
            frame = self._read_frame()
            end = time.perf_counter()
            if frame is None:
                self._capture_failures += 1
                time.sleep(0.01)
                continue
            with self._condition:
                self._latest_frame = frame
                self._latest_frame_id += 1
                self._latest_captured_at = end
                self._latest_capture_duration_ms = (end - start) * 1000.0
                self._capture_count += 1
                self._capture_last_at = end
                self._condition.notify_all()

    def _read_frame(self) -> np.ndarray | None:  # pragma: no cover - hardware-specific
        if self.picam is not None:
            frame = self.picam.capture_array()
            if frame is None:
                return None
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if self.cap is not None:
            ok, frame = self.cap.read()
            return frame if ok else None
        return None

    def wait_for_next_frame(self, last_frame_id: int, timeout_s: float = 5.0) -> FramePacket | None:
        deadline = time.perf_counter() + timeout_s
        with self._condition:
            while self._running and self._latest_frame_id <= last_frame_id:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return None
                self._condition.wait(timeout=remaining)
            if self._latest_frame is None or self._latest_frame_id <= last_frame_id:
                return None
            return FramePacket(
                frame=self._latest_frame.copy(),
                frame_id=self._latest_frame_id,
                captured_at=self._latest_captured_at,
                capture_duration_ms=self._latest_capture_duration_ms,
            )

    def counter_snapshot(self) -> tuple[int, float]:
        with self._condition:
            return self._capture_count, time.perf_counter()

    def capture_fps_between(
        self,
        start_snapshot: tuple[int, float],
        end_snapshot: tuple[int, float],
    ) -> float | None:
        count_delta = end_snapshot[0] - start_snapshot[0]
        time_delta = end_snapshot[1] - start_snapshot[1]
        if count_delta < 0 or time_delta <= 0:
            return None
        return count_delta / time_delta

    def stats(self) -> dict[str, Any]:
        elapsed = None
        if self._capture_started_at and self._capture_last_at >= self._capture_started_at:
            elapsed = self._capture_last_at - self._capture_started_at
        return {
            "camera_backend": self.backend,
            "capture_count_total": self._capture_count,
            "capture_failures_total": self._capture_failures,
            "capture_elapsed_sec_total": elapsed,
            "capture_fps_total": (self._capture_count / elapsed) if elapsed and elapsed > 0 else None,
            "warning": self.warning,
        }

    def release(self) -> None:  # pragma: no cover - hardware-specific
        self._running = False
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.picam is not None:
            try:
                self.picam.stop()
                self.picam.close()
            except Exception:
                pass
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass


class ONNXBenchmarkDetector:
    """Direct ONNX Runtime detector with separated timing buckets."""

    def __init__(self, model_path: Path, imgsz: int, provider: str) -> None:
        self.model_path = model_path
        self.imgsz = imgsz
        self.requested_provider = provider
        self.available_providers = ort.get_available_providers()
        self.provider_warning: str | None = None
        self.session_options = ort.SessionOptions()
        self.session_options.intra_op_num_threads = min(4, os.cpu_count() or 4)
        try:
            self.session_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        except Exception:
            pass
        provider_list = self._select_providers(provider)
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=self.session_options,
            providers=provider_list,
        )
        self.input_meta = self.session.get_inputs()[0]
        self.output_meta = self.session.get_outputs()[0]
        self.input_name = self.input_meta.name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.actual_providers = self.session.get_providers()
        self.actual_provider = self.actual_providers[0] if self.actual_providers else "unknown"
        self.static_input_height = self._extract_dim(self.input_meta.shape, 2)
        self.static_input_width = self._extract_dim(self.input_meta.shape, 3)
        input_mismatch = (
            (self.static_input_height is not None and self.static_input_height != imgsz)
            or (self.static_input_width is not None and self.static_input_width != imgsz)
        )
        if input_mismatch:
            raise ValueError(
                f"Model input is fixed at {self.static_input_width}x{self.static_input_height}. "
                f"Received --imgsz {imgsz}."
            )
        output_channels = self._extract_dim(self.output_meta.shape, 1)
        self.num_classes = max((output_channels or 5) - 4, 1)
        self.postprocess_backend = (
            "ultralytics.non_max_suppression"
            if torch is not None and non_max_suppression is not None
            else "opencv.dnn.NMSBoxes"
        )

    def _select_providers(self, provider: str) -> list[str]:
        if provider == "xnnpack":
            if "XNNPACKExecutionProvider" in self.available_providers:
                return ["XNNPACKExecutionProvider", "CPUExecutionProvider"]
            self.provider_warning = (
                "Requested XNNPACKExecutionProvider is unavailable. Falling back to CPUExecutionProvider."
            )
            return ["CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    @staticmethod
    def _extract_dim(shape: list[Any] | tuple[Any, ...], index: int) -> int | None:
        try:
            value = shape[index]
        except (IndexError, TypeError):
            return None
        return int(value) if isinstance(value, int) else None

    def detect(
        self,
        frame: np.ndarray,
        conf: float,
        clahe_enabled: bool,
    ) -> dict[str, Any]:
        if frame is None:
            raise ValueError("Received an empty frame for detection.")

        preprocess_start = time.perf_counter()
        orig_h, orig_w = frame.shape[:2]
        resized = cv2.resize(frame, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        if clahe_enabled:
            resized = apply_clahe(resized, clip=DEFAULT_CLAHE_CLIP, grid=DEFAULT_CLAHE_GRID)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = np.transpose(rgb, (2, 0, 1))[None].astype(np.float32) / 255.0
        preprocess_end = time.perf_counter()

        inference_start = preprocess_end
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        inference_end = time.perf_counter()

        postprocess_start = inference_end
        boxes, confs = self._postprocess(outputs[0], orig_w=orig_w, orig_h=orig_h, conf=conf)
        postprocess_end = time.perf_counter()

        preprocess_ms = (preprocess_end - preprocess_start) * 1000.0
        inference_ms = (inference_end - inference_start) * 1000.0
        postprocess_ms = (postprocess_end - postprocess_start) * 1000.0
        return {
            "boxes": boxes,
            "confs": confs,
            "num_detections": len(boxes),
            "preprocess_ms": preprocess_ms,
            "inference_ms": inference_ms,
            "postprocess_ms": postprocess_ms,
            "pipeline_ms": preprocess_ms + inference_ms + postprocess_ms,
        }

    def _postprocess(
        self,
        raw_output: np.ndarray,
        orig_w: int,
        orig_h: int,
        conf: float,
    ) -> tuple[list[list[int]], list[float]]:
        if torch is not None and non_max_suppression is not None:
            preds = torch.from_numpy(np.asarray(raw_output))
            detections = non_max_suppression(
                preds,
                conf_thres=conf,
                iou_thres=NMS_IOU_THRESHOLD,
                max_det=300,
                nc=self.num_classes,
            )[0]
            if detections is None or len(detections) == 0:
                return [], []
            det_array = detections.cpu().numpy()
            return self._scale_boxes(det_array[:, :4], det_array[:, 4], orig_w=orig_w, orig_h=orig_h)
        return self._manual_postprocess(raw_output, orig_w=orig_w, orig_h=orig_h, conf=conf)

    def _manual_postprocess(
        self,
        raw_output: np.ndarray,
        orig_w: int,
        orig_h: int,
        conf: float,
    ) -> tuple[list[list[int]], list[float]]:
        predictions = np.asarray(raw_output)[0].T
        if predictions.size == 0:
            return [], []

        if predictions.shape[1] <= 5:
            confidences = predictions[:, 4]
        else:
            confidences = np.max(predictions[:, 4:], axis=1)

        mask = confidences >= conf
        predictions = predictions[mask]
        confidences = confidences[mask]
        if predictions.size == 0:
            return [], []

        xywh = predictions[:, :4]
        xyxy = np.empty_like(xywh)
        xyxy[:, 0] = xywh[:, 0] - (xywh[:, 2] / 2.0)
        xyxy[:, 1] = xywh[:, 1] - (xywh[:, 3] / 2.0)
        xyxy[:, 2] = xywh[:, 0] + (xywh[:, 2] / 2.0)
        xyxy[:, 3] = xywh[:, 1] + (xywh[:, 3] / 2.0)

        nms_boxes = [
            [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            for x1, y1, x2, y2 in xyxy
        ]
        indices = cv2.dnn.NMSBoxes(nms_boxes, confidences.tolist(), conf, NMS_IOU_THRESHOLD)
        if len(indices) == 0:
            return [], []
        selected = np.asarray(indices).reshape(-1)
        selected_boxes = xyxy[selected]
        selected_confs = confidences[selected]
        return self._scale_boxes(selected_boxes, selected_confs, orig_w=orig_w, orig_h=orig_h)

    def _scale_boxes(
        self,
        boxes_xyxy: np.ndarray,
        confs: np.ndarray,
        orig_w: int,
        orig_h: int,
    ) -> tuple[list[list[int]], list[float]]:
        x_scale = orig_w / float(self.imgsz)
        y_scale = orig_h / float(self.imgsz)
        scaled_boxes: list[list[int]] = []
        scaled_confs: list[float] = []

        for box, conf_value in zip(boxes_xyxy, confs):
            x1 = int(round(box[0] * x_scale))
            y1 = int(round(box[1] * y_scale))
            x2 = int(round(box[2] * x_scale))
            y2 = int(round(box[3] * y_scale))

            x1 = max(0, min(orig_w - 1, x1))
            y1 = max(0, min(orig_h - 1, y1))
            x2 = max(0, min(orig_w - 1, x2))
            y2 = max(0, min(orig_h - 1, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            scaled_boxes.append([x1, y1, x2, y2])
            scaled_confs.append(float(conf_value))

        return scaled_boxes, scaled_confs


def collect_runtime_metadata(
    args: argparse.Namespace,
    detector: ONNXBenchmarkDetector,
    pre_benchmark_temp_c: float | None,
    pre_benchmark_throttled: dict[str, Any] | None,
) -> dict[str, Any]:
    os_release = parse_os_release()
    rpi_model = detect_raspberry_pi_model()
    cpu_model = detect_cpu_model()
    ram_mb = detect_ram_mb()
    metadata = {
        "timestamp_local": now_local_iso(),
        "timestamp_utc": now_utc_iso(),
        "hostname": socket.gethostname(),
        "python_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "platform_machine": platform.machine(),
        "os_pretty_name": os_release.get("PRETTY_NAME"),
        "os_name": os_release.get("NAME"),
        "os_version_id": os_release.get("VERSION_ID"),
        "raspberry_pi_model": rpi_model,
        "is_raspberry_pi_5": detect_is_raspberry_pi_5(rpi_model, cpu_model),
        "cpu_model": cpu_model,
        "cpu_count_logical": os.cpu_count(),
        "ram_mb": ram_mb,
        "onnxruntime_version": ort.__version__,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "torch_version": getattr(torch, "__version__", None),
        "ultralytics_nms_backend": detector.postprocess_backend,
        "ultralytics_nms_available": bool(torch is not None and non_max_suppression is not None),
        "ultralytics_nms_import_error": ULTRALYTICS_NMS_IMPORT_ERROR,
        "model_path": str(detector.model_path.resolve()),
        "model_file_size_bytes": detector.model_path.stat().st_size,
        "input_size": args.imgsz,
        "confidence_threshold": args.conf,
        "clahe": args.clahe,
        "clahe_clip_limit": DEFAULT_CLAHE_CLIP,
        "clahe_grid": list(DEFAULT_CLAHE_GRID),
        "execution_providers_available": detector.available_providers,
        "execution_provider_requested": args.provider,
        "execution_provider_used": detector.actual_provider,
        "execution_providers_used_chain": detector.actual_providers,
        "provider_warning": detector.provider_warning,
        "onnx_intra_op_num_threads": detector.session_options.intra_op_num_threads,
        "onnx_allow_spinning": "0",
        "warmup_frames": args.num_warmup,
        "benchmark_frames": args.num_runs,
        "benchmark_mode": args.mode,
        "preprocessing_included": True,
        "postprocessing_nms_included": True,
        "camera_capture_included": args.mode == "live_camera",
        "disk_image_read_included_in_end_to_end": args.mode == "image_folder",
        "pre_benchmark_cpu_temperature_c": pre_benchmark_temp_c,
        "pre_benchmark_throttled_status": pre_benchmark_throttled,
        "cli_command": " ".join(sys.argv),
    }
    return metadata


def benchmark_image_folder(
    detector: ONNXBenchmarkDetector,
    image_dir: Path,
    num_warmup: int,
    num_runs: int,
    conf: float,
    clahe_enabled: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    image_files = list_image_files(image_dir)
    image_iter = itertools.cycle(image_files)

    print(f"Image-folder benchmark using {len(image_files)} images from {image_dir}")

    for warmup_index in range(1, num_warmup + 1):
        path = next(image_iter)
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Failed to load warmup image: {path}")
        detector.detect(frame, conf=conf, clahe_enabled=clahe_enabled)
        progress_tick("warmup", warmup_index, num_warmup)

    records: list[dict[str, Any]] = []
    wall_start = time.perf_counter()
    for run_index in range(1, num_runs + 1):
        path = next(image_iter)
        start = time.perf_counter()
        frame = cv2.imread(str(path))
        after_read = time.perf_counter()
        if frame is None:
            raise RuntimeError(f"Failed to load benchmark image: {path}")
        detection = detector.detect(frame, conf=conf, clahe_enabled=clahe_enabled)
        end = time.perf_counter()

        image_read_ms = (after_read - start) * 1000.0
        end_to_end_ms = (end - start) * 1000.0
        records.append(
            {
                "run_index": run_index,
                "mode": "image_folder",
                "source_id": path.name,
                "source_path": str(path.resolve()),
                "frame_id": None,
                "image_read_ms": image_read_ms,
                "capture_wait_ms": None,
                "camera_capture_duration_ms": None,
                "frame_age_ms": None,
                "preprocess_ms": detection["preprocess_ms"],
                "inference_ms": detection["inference_ms"],
                "postprocess_ms": detection["postprocess_ms"],
                "pipeline_ms": detection["pipeline_ms"],
                "end_to_end_ms": end_to_end_ms,
                "num_detections": detection["num_detections"],
                "dropped_frames": 0,
            }
        )
        progress_tick("benchmark", run_index, num_runs)
    wall_end = time.perf_counter()

    mode_info = {
        "image_dir": str(image_dir.resolve()),
        "image_count_available": len(image_files),
        "wall_clock_elapsed_sec": wall_end - wall_start,
        "primary_metric_scope": "preprocessing + inference + postprocessing/NMS",
    }
    return records, mode_info


def benchmark_live_camera(
    detector: ONNXBenchmarkDetector,
    num_warmup: int,
    num_runs: int,
    conf: float,
    clahe_enabled: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    camera = BenchmarkCamera(
        width=DEFAULT_CAMERA_WIDTH,
        height=DEFAULT_CAMERA_HEIGHT,
        fps=DEFAULT_CAMERA_FPS,
    ).start()

    try:
        last_frame_id = 0
        print(f"Live-camera benchmark using backend: {camera.backend}")

        for warmup_index in range(1, num_warmup + 1):
            packet = camera.wait_for_next_frame(last_frame_id, timeout_s=5.0)
            if packet is None:
                raise RuntimeError("Timed out waiting for a warmup camera frame.")
            last_frame_id = packet.frame_id
            detector.detect(packet.frame, conf=conf, clahe_enabled=clahe_enabled)
            progress_tick("warmup", warmup_index, num_warmup)

        records: list[dict[str, Any]] = []
        dropped_total = 0
        camera_start_snapshot = camera.counter_snapshot()
        wall_start = time.perf_counter()

        for run_index in range(1, num_runs + 1):
            iteration_start = time.perf_counter()
            packet = camera.wait_for_next_frame(last_frame_id, timeout_s=5.0)
            wait_end = time.perf_counter()
            if packet is None:
                raise RuntimeError("Timed out waiting for a benchmark camera frame.")

            dropped_frames = max(packet.frame_id - last_frame_id - 1, 0) if last_frame_id else 0
            dropped_total += dropped_frames
            last_frame_id = packet.frame_id

            capture_wait_ms = (wait_end - iteration_start) * 1000.0
            frame_age_ms = max(0.0, (wait_end - packet.captured_at) * 1000.0)

            detection = detector.detect(packet.frame, conf=conf, clahe_enabled=clahe_enabled)
            iteration_end = time.perf_counter()

            records.append(
                {
                    "run_index": run_index,
                    "mode": "live_camera",
                    "source_id": f"frame_{packet.frame_id}",
                    "source_path": None,
                    "frame_id": packet.frame_id,
                    "image_read_ms": None,
                    "capture_wait_ms": capture_wait_ms,
                    "camera_capture_duration_ms": packet.capture_duration_ms,
                    "frame_age_ms": frame_age_ms,
                    "preprocess_ms": detection["preprocess_ms"],
                    "inference_ms": detection["inference_ms"],
                    "postprocess_ms": detection["postprocess_ms"],
                    "pipeline_ms": detection["pipeline_ms"],
                    "end_to_end_ms": (iteration_end - iteration_start) * 1000.0,
                    "num_detections": detection["num_detections"],
                    "dropped_frames": dropped_frames,
                }
            )
            progress_tick("benchmark", run_index, num_runs)

        wall_end = time.perf_counter()
        camera_end_snapshot = camera.counter_snapshot()
        camera_stats = camera.stats()
        mode_info = {
            "camera_backend": camera.backend,
            "camera_warning": camera.warning,
            "camera_capture_width": DEFAULT_CAMERA_WIDTH,
            "camera_capture_height": DEFAULT_CAMERA_HEIGHT,
            "camera_target_fps": DEFAULT_CAMERA_FPS,
            "camera_capture_fps_observed": camera.capture_fps_between(camera_start_snapshot, camera_end_snapshot),
            "camera_capture_failures_total": camera_stats["capture_failures_total"],
            "dropped_frames_total": dropped_total,
            "wall_clock_elapsed_sec": wall_end - wall_start,
            "primary_metric_scope": "full live-camera loop (capture wait + preprocessing + inference + postprocessing/NMS)",
        }
        return records, mode_info
    finally:
        camera.release()


def build_aggregate_metrics(
    records: list[dict[str, Any]],
    mode: str,
    wall_clock_elapsed_sec: float | None,
) -> dict[str, Any]:
    preprocess_ms = [float(row["preprocess_ms"]) for row in records]
    inference_ms = [float(row["inference_ms"]) for row in records]
    postprocess_ms = [float(row["postprocess_ms"]) for row in records]
    pipeline_ms = [float(row["pipeline_ms"]) for row in records]
    end_to_end_ms = [float(row["end_to_end_ms"]) for row in records]
    image_read_ms = [float(row["image_read_ms"]) for row in records if row["image_read_ms"] is not None]
    capture_wait_ms = [float(row["capture_wait_ms"]) for row in records if row["capture_wait_ms"] is not None]
    capture_duration_ms = [
        float(row["camera_capture_duration_ms"])
        for row in records
        if row["camera_capture_duration_ms"] is not None
    ]
    frame_age_ms = [float(row["frame_age_ms"]) for row in records if row["frame_age_ms"] is not None]
    dropped_frames_total = int(sum(int(row["dropped_frames"]) for row in records))

    primary_latency_ms = pipeline_ms if mode == "image_folder" else end_to_end_ms
    primary_scope = (
        "preprocessing + inference + postprocessing/NMS"
        if mode == "image_folder"
        else "full live-camera loop"
    )

    effective_end_to_end_fps = (
        (len(records) / wall_clock_elapsed_sec)
        if wall_clock_elapsed_sec and wall_clock_elapsed_sec > 0
        else None
    )

    return {
        "benchmark_frames_completed": len(records),
        "dropped_frames_total": dropped_frames_total,
        "wall_clock_elapsed_sec": wall_clock_elapsed_sec,
        "wall_clock_throughput_fps": effective_end_to_end_fps,
        "primary_metric_scope": primary_scope,
        "mean_latency_ms_per_frame": summarize_series(primary_latency_ms)["mean"] if primary_latency_ms else None,
        "median_latency_ms_per_frame": summarize_series(primary_latency_ms)["median"] if primary_latency_ms else None,
        "p90_latency_ms_per_frame": summarize_series(primary_latency_ms)["p90"] if primary_latency_ms else None,
        "p95_latency_ms_per_frame": summarize_series(primary_latency_ms)["p95"] if primary_latency_ms else None,
        "min_latency_ms_per_frame": summarize_series(primary_latency_ms)["min"] if primary_latency_ms else None,
        "max_latency_ms_per_frame": summarize_series(primary_latency_ms)["max"] if primary_latency_ms else None,
        "latency_std_ms_per_frame": summarize_series(primary_latency_ms)["std"] if primary_latency_ms else None,
        "mean_fps": summarize_series(fps_from_latency_ms(primary_latency_ms))["mean"]
        if primary_latency_ms
        else None,
        "median_fps": summarize_series(fps_from_latency_ms(primary_latency_ms))["median"]
        if primary_latency_ms
        else None,
        "effective_end_to_end_fps_live_camera": effective_end_to_end_fps if mode == "live_camera" else None,
        "preprocess_time_ms": summarize_series(preprocess_ms),
        "inference_time_ms": summarize_series(inference_ms),
        "postprocess_time_ms": summarize_series(postprocess_ms),
        "pipeline_latency_ms": summarize_series(pipeline_ms),
        "pipeline_fps": summarize_series(fps_from_latency_ms(pipeline_ms)),
        "end_to_end_latency_ms": summarize_series(end_to_end_ms),
        "end_to_end_fps": summarize_series(fps_from_latency_ms(end_to_end_ms)),
        "image_read_time_ms": summarize_series(image_read_ms),
        "capture_wait_time_ms": summarize_series(capture_wait_ms),
        "camera_capture_duration_ms": summarize_series(capture_duration_ms),
        "frame_age_ms": summarize_series(frame_age_ms),
    }


def resolve_output_paths(args: argparse.Namespace, slug: str) -> dict[str, Path | None]:
    return {
        "json": SCRIPT_DIR / f"benchmark_result_{slug}.json",
        "csv": SCRIPT_DIR / f"benchmark_result_{slug}.csv",
        "summary_md": SCRIPT_DIR / f"benchmark_summary_{slug}.md",
        "json_alias": Path(args.output_json) if args.output_json else None,
        "csv_alias": Path(args.output_csv) if args.output_csv else None,
        "summary_md_alias": Path(args.save_summary_md) if args.save_summary_md else None,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    fieldnames = [
        "run_index",
        "mode",
        "source_id",
        "source_path",
        "frame_id",
        "image_read_ms",
        "capture_wait_ms",
        "camera_capture_duration_ms",
        "frame_age_ms",
        "preprocess_ms",
        "inference_ms",
        "postprocess_ms",
        "pipeline_ms",
        "end_to_end_ms",
        "num_detections",
        "dropped_frames",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def summary_statement(is_raspberry_pi_5: bool) -> str:
    if is_raspberry_pi_5:
        return "This benchmark was executed on Raspberry Pi 5."
    return (
        "WARNING: This benchmark was not executed on Raspberry Pi 5 and must not be used as the embedded "
        "deployment result."
    )


def build_summary_markdown(
    metadata: dict[str, Any],
    mode_info: dict[str, Any],
    aggregate: dict[str, Any],
    outputs: dict[str, Path],
) -> str:
    lines: list[str] = []
    lines.append("# Raspberry Pi 5 ONNX Benchmark Summary")
    lines.append("")
    lines.append(summary_statement(bool(metadata["is_raspberry_pi_5"])))
    lines.append("")
    lines.append("## Benchmark Scope")
    lines.append("")
    lines.append(f"- Date/time (local): {metadata['timestamp_local']}")
    lines.append(f"- Benchmark mode: {metadata['benchmark_mode']}")
    lines.append(f"- Primary metric scope: {aggregate['primary_metric_scope']}")
    lines.append(f"- Model path: `{metadata['model_path']}`")
    lines.append(f"- Model file size (bytes): {metadata['model_file_size_bytes']}")
    lines.append(f"- Input size: {metadata['input_size']} x {metadata['input_size']}")
    lines.append(f"- Confidence threshold: {metadata['confidence_threshold']}")
    lines.append(f"- CLAHE: {metadata['clahe']}")
    lines.append(f"- Execution provider requested: {metadata['execution_provider_requested']}")
    lines.append(f"- Execution provider used: {metadata['execution_provider_used']}")
    lines.append(
        f"- Available execution providers: {', '.join(metadata['execution_providers_available']) or 'N/A'}"
    )
    lines.append(f"- Warmup frames: {metadata['warmup_frames']}")
    lines.append(f"- Benchmark frames: {metadata['benchmark_frames']}")
    lines.append(f"- Preprocessing included: {bool_to_yes_no(metadata['preprocessing_included'])}")
    lines.append(
        f"- Postprocessing/NMS included: {bool_to_yes_no(metadata['postprocessing_nms_included'])}"
    )
    lines.append(f"- Camera capture included: {bool_to_yes_no(metadata['camera_capture_included'])}")
    lines.append("")
    lines.append("## Key Results")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Mean latency (ms/frame) | {format_number(aggregate['mean_latency_ms_per_frame'])} |")
    lines.append(f"| Median latency (ms/frame) | {format_number(aggregate['median_latency_ms_per_frame'])} |")
    lines.append(f"| p90 latency (ms/frame) | {format_number(aggregate['p90_latency_ms_per_frame'])} |")
    lines.append(f"| p95 latency (ms/frame) | {format_number(aggregate['p95_latency_ms_per_frame'])} |")
    lines.append(f"| Min latency (ms/frame) | {format_number(aggregate['min_latency_ms_per_frame'])} |")
    lines.append(f"| Max latency (ms/frame) | {format_number(aggregate['max_latency_ms_per_frame'])} |")
    lines.append(f"| Std. dev. latency (ms/frame) | {format_number(aggregate['latency_std_ms_per_frame'])} |")
    lines.append(f"| Mean FPS | {format_number(aggregate['mean_fps'])} |")
    lines.append(f"| Median FPS | {format_number(aggregate['median_fps'])} |")
    lines.append(
        f"| Effective end-to-end FPS (live camera) | {format_number(aggregate['effective_end_to_end_fps_live_camera'])} |"
    )
    lines.append(
        f"| Mean preprocessing time (ms) | {format_number((aggregate['preprocess_time_ms'] or {}).get('mean'))} |"
    )
    lines.append(
        f"| Mean inference time (ms) | {format_number((aggregate['inference_time_ms'] or {}).get('mean'))} |"
    )
    lines.append(
        f"| Mean postprocessing/NMS time (ms) | {format_number((aggregate['postprocess_time_ms'] or {}).get('mean'))} |"
    )
    lines.append(f"| Dropped frames | {aggregate['dropped_frames_total']} |")
    if mode_info.get("camera_capture_fps_observed") is not None:
        lines.append(
            f"| Observed camera capture FPS | {format_number(mode_info['camera_capture_fps_observed'])} |"
        )
    lines.append("")
    lines.append("## Platform Metadata")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Hostname | `{metadata['hostname']}` |")
    lines.append(f"| Platform | `{metadata['platform']}` |")
    lines.append(f"| OS | `{metadata.get('os_pretty_name') or 'N/A'}` |")
    lines.append(f"| Raspberry Pi model | `{metadata.get('raspberry_pi_model') or 'N/A'}` |")
    lines.append(f"| CPU model | `{metadata.get('cpu_model') or 'N/A'}` |")
    lines.append(f"| RAM (MB) | `{metadata.get('ram_mb') or 'N/A'}` |")
    lines.append(f"| Python | `{metadata['python_version']}` |")
    lines.append(f"| ONNX Runtime | `{metadata['onnxruntime_version']}` |")
    lines.append(f"| OpenCV | `{metadata['opencv_version']}` |")
    lines.append(f"| NumPy | `{metadata['numpy_version']}` |")
    lines.append(
        f"| CPU temp before (C) | `{format_number(metadata.get('pre_benchmark_cpu_temperature_c'))}` |"
    )
    lines.append(
        f"| CPU temp after (C) | `{format_number(metadata.get('post_benchmark_cpu_temperature_c'))}` |"
    )
    lines.append(
        "| Throttled before | "
        f"`{(metadata.get('pre_benchmark_throttled_status') or {}).get('raw', 'N/A')}` |"
    )
    lines.append(
        "| Throttled after | "
        f"`{(metadata.get('post_benchmark_throttled_status') or {}).get('raw', 'N/A')}` |"
    )
    if mode_info.get("camera_backend"):
        lines.append(f"| Camera backend | `{mode_info['camera_backend']}` |")
    lines.append("")
    lines.append("## Artifact Files")
    lines.append("")
    lines.append(f"- JSON: `{outputs['json']}`")
    lines.append(f"- CSV: `{outputs['csv']}`")
    lines.append(f"- Summary Markdown: `{outputs['summary_md']}`")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- `Mean FPS` and `Mean latency` in this summary refer only to the declared primary metric scope above."
    )
    lines.append(
        "- `Pipeline` metrics isolate preprocessing + ONNX inference + postprocessing/NMS."
    )
    lines.append(
        "- `Effective end-to-end FPS (live camera)` is a wall-clock throughput figure and should not be confused with dashboard streaming FPS."
    )
    if metadata.get("provider_warning"):
        lines.append(f"- Provider warning: {metadata['provider_warning']}")
    if mode_info.get("camera_warning"):
        lines.append(f"- Camera warning: {mode_info['camera_warning']}")
    return "\n".join(lines) + "\n"


def write_summary_markdown(path: Path, content: str) -> None:
    ensure_parent(path)
    path.write_text(content, encoding="utf-8")


def maybe_write_alias(path: Path | None, content_writer: Any, source_payload: Any) -> None:
    if path is None:
        return
    content_writer(path, source_payload)


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    if model_path.suffix.lower() != ".onnx":
        raise ValueError("The benchmark script expects an ONNX model file.")
    if args.num_warmup < 0 or args.num_runs <= 0:
        raise ValueError("--num-warmup must be >= 0 and --num-runs must be > 0.")
    if not (0.0 < args.conf <= 1.0):
        raise ValueError("--conf must be in the range (0, 1].")
    if args.mode == "image_folder" and not args.image_dir:
        raise ValueError("--image-dir is required when --mode image_folder is selected.")

    slug = timestamp_slug()
    outputs = resolve_output_paths(args, slug)
    pre_temp = read_cpu_temperature_c()
    pre_throttled = read_throttled_status()

    detector = ONNXBenchmarkDetector(model_path=model_path, imgsz=args.imgsz, provider=args.provider)
    metadata = collect_runtime_metadata(args, detector, pre_temp, pre_throttled)

    clahe_enabled = on_off_to_bool(args.clahe)
    if args.mode == "image_folder":
        records, mode_info = benchmark_image_folder(
            detector=detector,
            image_dir=Path(args.image_dir),
            num_warmup=args.num_warmup,
            num_runs=args.num_runs,
            conf=args.conf,
            clahe_enabled=clahe_enabled,
        )
    else:
        records, mode_info = benchmark_live_camera(
            detector=detector,
            num_warmup=args.num_warmup,
            num_runs=args.num_runs,
            conf=args.conf,
            clahe_enabled=clahe_enabled,
        )

    metadata["post_benchmark_cpu_temperature_c"] = read_cpu_temperature_c()
    metadata["post_benchmark_throttled_status"] = read_throttled_status()

    aggregate = build_aggregate_metrics(
        records=records,
        mode=args.mode,
        wall_clock_elapsed_sec=mode_info.get("wall_clock_elapsed_sec"),
    )
    payload = {
        "metadata": metadata,
        "mode_info": mode_info,
        "aggregate_metrics": aggregate,
        "per_frame_results": records,
    }

    write_json(outputs["json"], payload)
    maybe_write_alias(outputs["json_alias"], write_json, payload)

    write_csv(outputs["csv"], records)
    maybe_write_alias(outputs["csv_alias"], write_csv, records)

    summary_content = build_summary_markdown(metadata, mode_info, aggregate, outputs)
    write_summary_markdown(outputs["summary_md"], summary_content)
    maybe_write_alias(outputs["summary_md_alias"], write_summary_markdown, summary_content)

    print(summary_statement(bool(metadata["is_raspberry_pi_5"])))
    print(f"JSON saved to: {outputs['json']}")
    print(f"CSV saved to: {outputs['csv']}")
    print(f"Summary saved to: {outputs['summary_md']}")
    if outputs["json_alias"] and outputs["json_alias"] != outputs["json"]:
        print(f"JSON alias saved to: {outputs['json_alias']}")
    if outputs["csv_alias"] and outputs["csv_alias"] != outputs["csv"]:
        print(f"CSV alias saved to: {outputs['csv_alias']}")
    if outputs["summary_md_alias"] and outputs["summary_md_alias"] != outputs["summary_md"]:
        print(f"Summary alias saved to: {outputs['summary_md_alias']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Benchmark interrupted by user.")
        raise SystemExit(130)
