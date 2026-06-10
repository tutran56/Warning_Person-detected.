from __future__ import annotations

import csv
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles


# ============================================================
# 1. CONFIG WEBCAM
# ============================================================

DEFAULT_CAMERA_SOURCE = "0"

STREAM_WIDTH = 640
STREAM_HEIGHT = 360

# Chạy MobileNet-SSD mỗi N frame để giảm tải CPU.
DETECTION_EVERY_N_FRAMES = 3

# Giữ kết quả detect trong thời gian ngắn để tránh nhấp nháy.
DETECTION_TTL_SECONDS = 0.8

# Ngưỡng tin cậy của MobileNet-SSD.
SSD_CONFIDENCE_THRESHOLD = 0.45

# MobileNet-SSD input size.
SSD_INPUT_SIZE = (300, 300)


# ============================================================
# 2. FOLDER CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent
INDEX_HTML = ROOT / "index.html"

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw_images"
PROCESSED_DIR = DATA_DIR / "processed_images"
VIDEO_DIR = DATA_DIR / "videos"

OUTPUT_DIR = ROOT / "outputs"
METADATA_CSV = OUTPUT_DIR / "image_metadata.csv"
EVENT_CSV = OUTPUT_DIR / "image_event_log.csv"


# ============================================================
# 3. MOBILENET-SSD MODEL PATH
# ============================================================
# SỬA CHÍNH Ở ĐÂY:
# Dùng trực tiếp đường dẫn tuyệt đối, không nối với MODEL_DIR nữa.

SSD_PROTO = Path(
    "/Users/quang/Downloads/lab6_cv_as_iot_sensor/MobileNetSSD_deploy.prototxt"
)

SSD_MODEL = Path(
    "/Users/quang/Downloads/lab6_cv_as_iot_sensor/mobilenet_iter_73000.caffemodel"
)

# Nếu file .caffemodel của bạn tên khác, ví dụ:
# MobileNetSSD_deploy.caffemodel
# thì sửa SSD_MODEL thành:
# SSD_MODEL = Path("/Users/quang/Downloads/lab6_cv_as_iot_sensor/MobileNetSSD_deploy.caffemodel")


for folder in [RAW_DIR, PROCESSED_DIR, VIDEO_DIR, OUTPUT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


METADATA_FIELDS = [
    "image_id",
    "device_id",
    "timestamp",
    "source_type",
    "image_path",
    "processed_path",
    "width",
    "height",
    "brightness",
    "processing_status",
    "processing_time_ms",
    "note",
]

EVENT_FIELDS = [
    "event_id",
    "image_id",
    "timestamp",
    "event_type",
    "score",
    "severity",
    "explanation",
    "action_hint",
]


# ============================================================
# 4. MOBILENET-SSD CONFIG
# ============================================================

SSD_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

# Detect tất cả class vật thể mà MobileNet-SSD hỗ trợ, bao gồm person.
# Nếu chỉ muốn detect người, đổi thành: {"person"}
ALLOWED_DETECTION_CLASSES: Set[str] = set(SSD_CLASSES) - {"background"}

SSD_NET: Optional[cv2.dnn_Net] = None
SSD_LOAD_ERROR = ""
SSD_LOAD_ATTEMPTED = False
SSD_LOCK = threading.Lock()


# ============================================================
# 5. UTILS
# ============================================================

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def normalize_source(source: str) -> str:
    source = str(source or "").strip()
    return source if source else DEFAULT_CAMERA_SOURCE


def parse_source(source: str) -> Any:
    source = normalize_source(source)

    if source.isdigit():
        return int(source)

    return source


def path_display(path: Path) -> str:
    """
    Hiển thị path an toàn.
    Nếu path nằm trong project thì hiển thị dạng relative.
    Nếu path là absolute bên ngoài project, hiển thị nguyên path.
    """
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path)


def append_csv(path: Path, fields: List[str], row: Dict[str, Any]) -> None:
    need_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)

        if need_header:
            writer.writeheader()

        writer.writerow({key: row.get(key, "") for key in fields})


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []

    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def relative_url(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None

    try:
        rel = path.resolve().relative_to(ROOT.resolve())
        return f"/files/{rel.as_posix()}"
    except Exception:
        return None


def frame_to_jpeg_bytes(frame: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".jpg", frame)

    if not ok:
        raise RuntimeError("Không encode được frame thành JPEG")

    return buffer.tobytes()


def compute_brightness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def simulated_frame(counter: int = 0) -> np.ndarray:
    frame = np.full((STREAM_HEIGHT, STREAM_WIDTH, 3), 245, dtype=np.uint8)

    x = 30 + (counter * 10) % max(1, STREAM_WIDTH - 180)
    y = 85 + (counter * 5) % max(1, STREAM_HEIGHT - 160)

    cv2.rectangle(frame, (x, 120), (x + 130, 240), (40, 140, 240), -1)
    cv2.circle(frame, (STREAM_WIDTH - 110, y), 38, (80, 200, 120), -1)

    cv2.putText(
        frame,
        "SIMULATED - WEBCAM NOT CONNECTED",
        (25, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        2,
    )

    cv2.putText(
        frame,
        "MobileNet-SSD object detection app",
        (25, STREAM_HEIGHT - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
    )

    return frame


def get_ssd_net() -> Optional[cv2.dnn_Net]:
    """
    Load MobileNet-SSD từ file local.
    Không tự download nữa.
    """
    global SSD_NET, SSD_LOAD_ERROR, SSD_LOAD_ATTEMPTED

    if SSD_NET is not None:
        return SSD_NET

    if SSD_LOAD_ATTEMPTED and SSD_NET is None:
        return None

    with SSD_LOCK:
        if SSD_NET is not None:
            return SSD_NET

        SSD_LOAD_ATTEMPTED = True

        try:
            if not SSD_PROTO.exists():
                raise FileNotFoundError(
                    f"Không tìm thấy file prototxt: {SSD_PROTO}"
                )

            if not SSD_MODEL.exists():
                raise FileNotFoundError(
                    f"Không tìm thấy file caffemodel: {SSD_MODEL}"
                )

            SSD_NET = cv2.dnn.readNetFromCaffe(
                str(SSD_PROTO),
                str(SSD_MODEL),
            )

            SSD_LOAD_ERROR = ""
            print("MobileNet-SSD loaded successfully.")
            print(f"PROTO: {SSD_PROTO}")
            print(f"MODEL: {SSD_MODEL}")

            return SSD_NET

        except Exception as exc:
            SSD_LOAD_ERROR = f"Không load được MobileNet-SSD. Lỗi: {exc}"
            print(SSD_LOAD_ERROR)
            return None


# ============================================================
# 6. LOG IMAGE / EVENT
# ============================================================

def create_processed_contact_sheet(
    frame: np.ndarray,
    image_id: str,
) -> Tuple[Path, float, Dict[str, Any]]:
    start = time.perf_counter()

    resized = cv2.resize(frame, (320, 240))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    _, threshold = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    edges = cv2.Canny(gray, 80, 160)

    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    threshold_bgr = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)
    edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    def label(tile: np.ndarray, text: str) -> np.ndarray:
        canvas = tile.copy()

        cv2.rectangle(canvas, (0, 0), (320, 30), (255, 255, 255), -1)

        cv2.putText(
            canvas,
            text,
            (10, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 0),
            2,
        )

        return canvas

    top = np.hstack(
        [
            label(resized, "1. RESIZE"),
            label(gray_bgr, "2. GRAYSCALE"),
        ]
    )

    bottom = np.hstack(
        [
            label(threshold_bgr, "3. THRESHOLD"),
            label(edge_bgr, "4. EDGE"),
        ]
    )

    sheet = np.vstack([top, bottom])

    out_path = PROCESSED_DIR / f"{image_id}_processed_steps.jpg"
    cv2.imwrite(str(out_path), sheet)

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    stats = {
        "brightness": round(compute_brightness(frame), 2),
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
    }

    return out_path, elapsed_ms, stats


def add_event(
    image_id: str,
    event_type: str,
    score: Any,
    severity: str,
    explanation: str,
    action_hint: str,
) -> Dict[str, Any]:
    row = {
        "event_id": f"evt_{uuid.uuid4().hex[:10]}",
        "image_id": image_id,
        "timestamp": now_iso(),
        "event_type": event_type,
        "score": score,
        "severity": severity,
        "explanation": explanation,
        "action_hint": action_hint,
    }

    append_csv(EVENT_CSV, EVENT_FIELDS, row)

    return row


def log_image_pipeline(
    frame: np.ndarray,
    source_type: str,
    device_id: str,
    note: str = "",
) -> Dict[str, Any]:
    image_id = f"img_{uuid.uuid4().hex[:10]}"
    timestamp = now_iso()

    raw_path = RAW_DIR / f"{image_id}.jpg"
    cv2.imwrite(str(raw_path), frame)

    processed_path, processing_time_ms, stats = create_processed_contact_sheet(
        frame,
        image_id,
    )

    metadata_row = {
        "image_id": image_id,
        "device_id": device_id,
        "timestamp": timestamp,
        "source_type": source_type,
        "image_path": str(raw_path.relative_to(ROOT)),
        "processed_path": str(processed_path.relative_to(ROOT)),
        "width": stats["width"],
        "height": stats["height"],
        "brightness": stats["brightness"],
        "processing_status": "processed",
        "processing_time_ms": processing_time_ms,
        "note": note,
    }

    append_csv(METADATA_CSV, METADATA_FIELDS, metadata_row)

    return {
        "image_id": image_id,
        "metadata": metadata_row,
        "raw_image_url": relative_url(raw_path),
        "processed_image_url": relative_url(processed_path),
    }


# ============================================================
# 7. DETECTION FUNCTIONS - MOBILENET SSD
# ============================================================

def clamp(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(value, max_value))


def detect_objects_mobilenet_ssd(
    frame: np.ndarray,
    min_confidence: float = SSD_CONFIDENCE_THRESHOLD,
    allowed_classes: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    net = get_ssd_net()

    if net is None:
        return []

    h, w = frame.shape[:2]

    if h <= 0 or w <= 0:
        return []

    if allowed_classes is None:
        allowed_classes = ALLOWED_DETECTION_CLASSES

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, SSD_INPUT_SIZE),
        scalefactor=0.007843,
        size=SSD_INPUT_SIZE,
        mean=127.5,
    )

    with SSD_LOCK:
        net.setInput(blob)
        detections = net.forward()

    results: List[Dict[str, Any]] = []

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])

        if confidence < min_confidence:
            continue

        class_id = int(detections[0, 0, i, 1])

        if class_id < 0 or class_id >= len(SSD_CLASSES):
            continue

        label = SSD_CLASSES[class_id]

        if label not in allowed_classes:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype("int")

        x1 = clamp(int(x1), 0, w - 1)
        y1 = clamp(int(y1), 0, h - 1)
        x2 = clamp(int(x2), 0, w - 1)
        y2 = clamp(int(y2), 0, h - 1)

        if x2 <= x1 or y2 <= y1:
            continue

        results.append(
            {
                "label": label,
                "class_id": class_id,
                "confidence": round(confidence, 4),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "w": int(x2 - x1),
                "h": int(y2 - y1),
                "area": int((x2 - x1) * (y2 - y1)),
            }
        )

    return apply_nms(results)


def apply_nms(
    detections: List[Dict[str, Any]],
    score_threshold: float = SSD_CONFIDENCE_THRESHOLD,
    nms_threshold: float = 0.35,
) -> List[Dict[str, Any]]:
    if not detections:
        return []

    boxes = []
    scores = []

    for det in detections:
        boxes.append(
            [
                int(det["x1"]),
                int(det["y1"]),
                int(det["w"]),
                int(det["h"]),
            ]
        )
        scores.append(float(det["confidence"]))

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        score_threshold,
        nms_threshold,
    )

    if len(indices) == 0:
        return []

    kept: List[Dict[str, Any]] = []

    for idx in indices.flatten():
        kept.append(detections[int(idx)])

    kept.sort(key=lambda item: float(item["confidence"]), reverse=True)

    return kept


def draw_detections(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
) -> np.ndarray:
    output = frame.copy()

    for det in detections:
        x1 = int(det["x1"])
        y1 = int(det["y1"])
        x2 = int(det["x2"])
        y2 = int(det["y2"])

        label = str(det["label"])
        confidence = float(det["confidence"])

        if label == "person":
            color = (0, 0, 255)
        else:
            color = (0, 180, 0)

        # Chỉ vẽ bounding box khi MobileNet-SSD phát hiện người/vật thật.
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        text = f"{label}: {confidence:.2f}"
        text_y = max(22, y1 - 8)

        cv2.rectangle(
            output,
            (x1, text_y - 18),
            (min(x1 + 180, output.shape[1] - 1), text_y + 5),
            color,
            -1,
        )

        cv2.putText(
            output,
            text,
            (x1 + 4, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

    return output


# ============================================================
# 8. CAMERA WORKER
# ============================================================

class CameraWorker:
    def __init__(self, source: str):
        self.source = normalize_source(source)
        self.cap: Optional[cv2.VideoCapture] = None

        self.lock = threading.Lock()
        self.running = False
        self.thread: Optional[threading.Thread] = None

        self.frame: Optional[np.ndarray] = None
        self.annotated_frame: Optional[np.ndarray] = None

        self.counter = 0
        self.source_label = "INIT"
        self.last_error = ""

        self.fps = 0.0
        self._fps_counter = 0
        self._fps_time = time.perf_counter()

        self.detections: List[Dict[str, Any]] = []
        self.last_detection_time = 0.0

        self.person_count = 0
        self.object_count = 0
        self.person_detected = False
        self.object_detected = False
        self.any_detection = False

        self.event_type = "INIT"

        self.notification_active = False
        self.notification_text = ""

        # Giữ field cũ để index.html không lỗi.
        self.alert_active = False
        self.alert_message = ""

    def start(self) -> None:
        if self.running:
            return

        self.running = True

        self.thread = threading.Thread(
            target=self._loop,
            daemon=True,
        )

        self.thread.start()

    def _open(self) -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(parse_source(self.source))

        if not cap.isOpened():
            self.last_error = f"Không mở được webcam source={self.source}"
            return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, STREAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)

        return cap

    def _update_fps(self) -> None:
        self._fps_counter += 1

        now = time.perf_counter()
        elapsed = now - self._fps_time

        if elapsed >= 1.0:
            self.fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_time = now

    def _make_annotated(self, frame: np.ndarray) -> np.ndarray:
        output = draw_detections(frame, self.detections)

        if self.any_detection:
            status = (
                f"DETECTED | persons={self.person_count} "
                f"| objects={self.object_count}"
            )
            status_color = (0, 255, 255)
        else:
            status = "MONITORING - NO PERSON / OBJECT"
            status_color = (255, 255, 255)

        cv2.rectangle(
            output,
            (0, 0),
            (output.shape[1], 92),
            (20, 20, 20),
            -1,
        )

        cv2.putText(
            output,
            f"WEBCAM source={self.source}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            output,
            status,
            (10, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            status_color,
            2,
        )

        cv2.putText(
            output,
            f"detector=MobileNet-SSD | confidence>={SSD_CONFIDENCE_THRESHOLD}",
            (10, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        if SSD_LOAD_ERROR:
            cv2.putText(
                output,
                "MODEL ERROR - check /health",
                (output.shape[1] - 260, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
            )

        return output

    def _loop(self) -> None:
        self.cap = self._open()
        last_retry = time.perf_counter()

        while self.running:
            self._update_fps()

            if self.cap is None:
                raw = simulated_frame(self.counter)
                label = "SIMULATED_NO_WEBCAM"

                if time.perf_counter() - last_retry > 3:
                    self.cap = self._open()
                    last_retry = time.perf_counter()

            else:
                ok, raw = self.cap.read()

                if not ok or raw is None:
                    self.last_error = "Không đọc được frame từ webcam"

                    self.cap.release()
                    self.cap = None

                    raw = simulated_frame(self.counter)
                    label = "SIMULATED_AFTER_WEBCAM_ERROR"

                else:
                    label = "WEBCAM_LIVE"

            raw = cv2.resize(raw, (STREAM_WIDTH, STREAM_HEIGHT))

            now = time.perf_counter()

            if self.counter % DETECTION_EVERY_N_FRAMES == 0:
                current_detections = detect_objects_mobilenet_ssd(
                    raw,
                    min_confidence=SSD_CONFIDENCE_THRESHOLD,
                    allowed_classes=ALLOWED_DETECTION_CLASSES,
                )

                if current_detections:
                    self.detections = current_detections
                    self.last_detection_time = now

            if now - self.last_detection_time > DETECTION_TTL_SECONDS:
                self.detections = []

            self.person_count = sum(
                1 for item in self.detections if item["label"] == "person"
            )

            self.object_count = sum(
                1 for item in self.detections if item["label"] != "person"
            )

            self.person_detected = self.person_count > 0
            self.object_detected = self.object_count > 0
            self.any_detection = len(self.detections) > 0

            if self.person_detected:
                self.event_type = "PERSON_DETECTED"
                self.notification_active = True
                self.notification_text = "Phát hiện người bằng MobileNet-SSD"

            elif self.object_detected:
                self.event_type = "OBJECT_DETECTED"
                self.notification_active = True
                self.notification_text = "Phát hiện vật thể bằng MobileNet-SSD"

            else:
                self.event_type = "MONITORING"
                self.notification_active = False
                self.notification_text = "Không phát hiện người/vật"

            self.alert_active = False
            self.alert_message = ""

            annotated = self._make_annotated(raw)

            with self.lock:
                self.frame = raw.copy()
                self.annotated_frame = annotated.copy()
                self.source_label = label
                self.counter += 1

            time.sleep(0.03)

        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def get_frame(self) -> Tuple[np.ndarray, str, int]:
        with self.lock:
            if self.frame is None:
                return simulated_frame(0), "SIMULATED_NOT_READY", self.counter

            return self.frame.copy(), self.source_label, self.counter

    def get_annotated_frame(self) -> Tuple[np.ndarray, str, int]:
        with self.lock:
            if self.annotated_frame is None:
                return simulated_frame(0), "SIMULATED_NOT_READY", self.counter

            return self.annotated_frame.copy(), self.source_label, self.counter

    def state(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "source": self.source,
                "source_label": self.source_label,
                "frames_seen": self.counter,
                "fps": round(self.fps, 2),
                "has_frame": self.frame is not None,
                "last_error": self.last_error,

                "detector": "MobileNet-SSD",
                "model_proto": path_display(SSD_PROTO),
                "model_caffemodel": path_display(SSD_MODEL),
                "model_proto_exists": SSD_PROTO.exists(),
                "model_caffemodel_exists": SSD_MODEL.exists(),
                "model_ready": SSD_NET is not None,
                "model_error": SSD_LOAD_ERROR,
                "confidence_threshold": SSD_CONFIDENCE_THRESHOLD,

                "detections": self.detections,
                "detection_count": len(self.detections),
                "person_count": self.person_count,
                "object_count": self.object_count,
                "person_detected": self.person_detected,
                "object_detected": self.object_detected,
                "any_detection": self.any_detection,

                "event_type": self.event_type,

                "notification_active": self.notification_active,
                "notification_text": self.notification_text,

                # Field cũ để HTML cũ không bị lỗi.
                "motion_score_raw": 0,
                "motion_score_smooth": 0,
                "motion_detected": False,
                "motion_bbox": None,

                "face_count": 0,
                "faces": [],

                "persons": [
                    item for item in self.detections if item["label"] == "person"
                ],
                "human_detected": self.person_detected,
                "human_motion_detected": self.person_detected,
                "person_moving_detected": self.person_detected,

                "alert_active": False,
                "alert_message": "",
            }


CAMERA_WORKERS: Dict[str, CameraWorker] = {}


def get_camera_worker(source: str = DEFAULT_CAMERA_SOURCE) -> CameraWorker:
    src = normalize_source(source)

    if src not in CAMERA_WORKERS:
        CAMERA_WORKERS[src] = CameraWorker(src)
        CAMERA_WORKERS[src].start()

    return CAMERA_WORKERS[src]


def stream_frames(source: str = DEFAULT_CAMERA_SOURCE) -> Iterable[bytes]:
    worker = get_camera_worker(source)

    while True:
        frame, _, _ = worker.get_annotated_frame()

        jpg = frame_to_jpeg_bytes(frame)

        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"

        time.sleep(0.03)


# ============================================================
# 9. FASTAPI APP
# ============================================================

app = FastAPI(
    title="OpenCV MobileNet-SSD Object Detection",
    description="Webcam + FPS + MobileNet-SSD person/object detection.",
)

app.mount("/files", StaticFiles(directory=str(ROOT)), name="files")


@app.get("/")
def home() -> FileResponse:
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=404, detail="Không tìm thấy index.html")

    return FileResponse(INDEX_HTML)


@app.get("/dashboard")
def dashboard() -> RedirectResponse:
    return RedirectResponse("/")


@app.get("/health")
def health() -> Dict[str, Any]:
    worker = get_camera_worker(DEFAULT_CAMERA_SOURCE)

    return {
        "status": "ok",
        "mode": "webcam",
        "default_camera_source": DEFAULT_CAMERA_SOURCE,
        "stream_width": STREAM_WIDTH,
        "stream_height": STREAM_HEIGHT,
        "detector": "MobileNet-SSD Caffe via OpenCV DNN",
        "rule": "Chỉ vẽ bounding box khi MobileNet-SSD phát hiện person/object",
        "allowed_classes": sorted(ALLOWED_DETECTION_CLASSES),
        "model_proto": path_display(SSD_PROTO),
        "model_caffemodel": path_display(SSD_MODEL),
        "model_proto_exists": SSD_PROTO.exists(),
        "model_caffemodel_exists": SSD_MODEL.exists(),
        "model_ready": SSD_NET is not None,
        "model_error": SSD_LOAD_ERROR,
        "camera_state": worker.state(),
    }


@app.get("/camera-status")
def camera_status(
    source: str = Query(DEFAULT_CAMERA_SOURCE),
) -> Dict[str, Any]:
    worker = get_camera_worker(source)
    return worker.state()


@app.get("/video_feed")
def video_feed(
    source: str = Query(DEFAULT_CAMERA_SOURCE),
) -> StreamingResponse:
    return StreamingResponse(
        stream_frames(source),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/snapshot")
def snapshot(
    source: str = Query(DEFAULT_CAMERA_SOURCE),
) -> Dict[str, Any]:
    worker = get_camera_worker(source)
    frame, source_label, _ = worker.get_annotated_frame()
    state = worker.state()

    result = log_image_pipeline(
        frame=frame,
        source_type="webcam_snapshot",
        device_id=f"camera:{normalize_source(source)}",
        note=f"source_label={source_label}, event_type={state['event_type']}",
    )

    result["camera_state"] = state

    return result


@app.get("/motion-capture")
def motion_capture_endpoint(
    source: str = Query(DEFAULT_CAMERA_SOURCE),
) -> Dict[str, Any]:
    """
    Giữ endpoint cũ để index.html không lỗi.
    Bây giờ endpoint này dùng MobileNet-SSD, không dùng motion bbox nữa.
    """
    worker = get_camera_worker(source)
    frame, _, _ = worker.get_annotated_frame()
    state = worker.state()

    result = log_image_pipeline(
        frame=frame,
        source_type="mobilenet_ssd_capture",
        device_id=f"camera:{normalize_source(source)}",
        note=(
            f"event_type={state['event_type']}, "
            f"fps={state['fps']}, "
            f"detection_count={state['detection_count']}, "
            f"person_count={state['person_count']}, "
            f"object_count={state['object_count']}"
        ),
    )

    if state["person_detected"]:
        event_type = "PERSON_DETECTED"
        severity = "HIGH"
        explanation = "Phát hiện người bằng MobileNet-SSD."
        action_hint = "Thông báo: có người xuất hiện trước webcam."

    elif state["object_detected"]:
        event_type = "OBJECT_DETECTED"
        severity = "WARNING"
        explanation = "Phát hiện vật thể bằng MobileNet-SSD."
        action_hint = "Thông báo: có vật thể xuất hiện trước webcam."

    else:
        event_type = "NO_OBJECT_DETECTED"
        severity = "NORMAL"
        explanation = "Không phát hiện người/vật."
        action_hint = "Tiếp tục giám sát."

    event = add_event(
        image_id=result["image_id"],
        event_type=event_type,
        score=state["detection_count"],
        severity=severity,
        explanation=explanation,
        action_hint=action_hint,
    )

    result.update(
        {
            "motion_event": event,
            "camera_state": state,
            "fps": state["fps"],
            "event_type": state["event_type"],
            "notification_active": state["notification_active"],
            "notification_text": state["notification_text"],

            "detections": state["detections"],
            "detection_count": state["detection_count"],
            "person_detected": state["person_detected"],
            "object_detected": state["object_detected"],
            "person_count": state["person_count"],
            "object_count": state["object_count"],

            # Field cũ để HTML cũ không lỗi.
            "motion_detected": False,
            "person_moving_detected": state["person_detected"],
            "human_motion_detected": state["person_detected"],
            "human_detected": state["person_detected"],
            "alert_active": False,
            "alert_message": "",
        }
    )

    return result


@app.get("/face-motion-capture")
def face_motion_capture_endpoint(
    source: str = Query(DEFAULT_CAMERA_SOURCE),
) -> Dict[str, Any]:
    # Giữ endpoint cũ để HTML không lỗi.
    return motion_capture_endpoint(source=source)


@app.get("/detect-person")
def detect_person_endpoint(
    source: str = Query(DEFAULT_CAMERA_SOURCE),
    confidence: float = Query(SSD_CONFIDENCE_THRESHOLD, ge=0.0, le=1.0),
) -> Dict[str, Any]:
    worker = get_camera_worker(source)
    frame, _, _ = worker.get_frame()

    persons = detect_objects_mobilenet_ssd(
        frame,
        min_confidence=confidence,
        allowed_classes={"person"},
    )

    annotated = draw_detections(frame, persons)

    result = log_image_pipeline(
        frame=annotated,
        source_type="detect_person_mobilenet_ssd",
        device_id=f"camera:{normalize_source(source)}",
        note=f"persons={len(persons)}, confidence={confidence}",
    )

    event = add_event(
        image_id=result["image_id"],
        event_type="PERSON_DETECTED" if persons else "NO_PERSON_DETECTED",
        score=len(persons),
        severity="WARNING" if persons else "NORMAL",
        explanation=(
            f"Phát hiện {len(persons)} người bằng MobileNet-SSD."
            if persons
            else "Không phát hiện người."
        ),
        action_hint="Tăng ánh sáng hoặc giảm confidence nếu model khó phát hiện.",
    )

    result.update(
        {
            "person_event": event,
            "person_detected": len(persons) > 0,
            "person_count": len(persons),
            "persons": persons,
            "detections": persons,
            "model_error": SSD_LOAD_ERROR,
        }
    )

    return result


@app.get("/detect-objects")
def detect_objects_endpoint(
    source: str = Query(DEFAULT_CAMERA_SOURCE),
    confidence: float = Query(SSD_CONFIDENCE_THRESHOLD, ge=0.0, le=1.0),
) -> Dict[str, Any]:
    worker = get_camera_worker(source)
    frame, _, _ = worker.get_frame()

    detections = detect_objects_mobilenet_ssd(
        frame,
        min_confidence=confidence,
        allowed_classes=ALLOWED_DETECTION_CLASSES,
    )

    annotated = draw_detections(frame, detections)

    person_count = sum(1 for item in detections if item["label"] == "person")
    object_count = sum(1 for item in detections if item["label"] != "person")

    result = log_image_pipeline(
        frame=annotated,
        source_type="detect_objects_mobilenet_ssd",
        device_id=f"camera:{normalize_source(source)}",
        note=(
            f"detections={len(detections)}, "
            f"person_count={person_count}, "
            f"object_count={object_count}, "
            f"confidence={confidence}"
        ),
    )

    event = add_event(
        image_id=result["image_id"],
        event_type="OBJECTS_DETECTED" if detections else "NO_OBJECT_DETECTED",
        score=len(detections),
        severity="WARNING" if detections else "NORMAL",
        explanation=(
            f"Phát hiện {len(detections)} người/vật bằng MobileNet-SSD."
            if detections
            else "Không phát hiện người/vật."
        ),
        action_hint="Bounding box chỉ được vẽ khi model phát hiện class hợp lệ.",
    )

    result.update(
        {
            "object_event": event,
            "detections": detections,
            "detection_count": len(detections),
            "person_count": person_count,
            "object_count": object_count,
            "person_detected": person_count > 0,
            "object_detected": object_count > 0,
            "model_error": SSD_LOAD_ERROR,
        }
    )

    return result


@app.get("/detect-face")
def detect_face_endpoint(
    source: str = Query(DEFAULT_CAMERA_SOURCE),
) -> Dict[str, Any]:
    """
    Giữ endpoint cũ để HTML không lỗi.
    App này không dùng face detector nữa.
    """
    result = detect_objects_endpoint(source=source)

    result.update(
        {
            "face_count": 0,
            "faces": [],
            "message": "Face detector đã tắt. App đang dùng MobileNet-SSD để detect person/object.",
        }
    )

    return result


@app.get("/record-video")
def record_video(
    source: str = Query(DEFAULT_CAMERA_SOURCE),
    seconds: int = Query(5, ge=1, le=30),
) -> Dict[str, Any]:
    worker = get_camera_worker(source)

    seconds = max(1, min(int(seconds), 30))

    video_id = f"vid_{uuid.uuid4().hex[:10]}"
    out_path = VIDEO_DIR / f"{video_id}.mp4"

    fps = 20

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (STREAM_WIDTH, STREAM_HEIGHT),
    )

    frame_count = 0
    start = time.perf_counter()

    while time.perf_counter() - start < seconds:
        frame, _, _ = worker.get_annotated_frame()
        frame = cv2.resize(frame, (STREAM_WIDTH, STREAM_HEIGHT))

        writer.write(frame)

        frame_count += 1

        time.sleep(1.0 / fps)

    writer.release()

    event = add_event(
        image_id=video_id,
        event_type="VIDEO_RECORDED",
        score=frame_count,
        severity="NORMAL",
        explanation=f"Đã ghi video {seconds}s gồm {frame_count} frame.",
        action_hint="Dùng video để kiểm chứng bounding box MobileNet-SSD.",
    )

    return {
        "video_id": video_id,
        "video_path": str(out_path.relative_to(ROOT)),
        "video_url": relative_url(out_path),
        "frames": frame_count,
        "seconds": seconds,
        "event": event,
    }


@app.get("/metadata")
def metadata(
    limit: int = 20,
) -> Dict[str, Any]:
    rows = read_csv(METADATA_CSV)

    return {
        "count": len(rows),
        "items": rows[-limit:],
    }


@app.get("/events")
def events(
    limit: int = 20,
) -> Dict[str, Any]:
    rows = read_csv(EVENT_CSV)

    return {
        "count": len(rows),
        "items": rows[-limit:],
    }


@app.get("/latest")
def latest() -> Dict[str, Any]:
    meta_rows = read_csv(METADATA_CSV)
    event_rows = read_csv(EVENT_CSV)

    latest_meta = meta_rows[-1] if meta_rows else None

    raw_url = None
    processed_url = None

    if latest_meta:
        image_path = latest_meta.get("image_path", "")
        processed_path = latest_meta.get("processed_path", "")

        raw_url = relative_url(ROOT / image_path) if image_path else None
        processed_url = relative_url(ROOT / processed_path) if processed_path else None

    return {
        "latest_metadata": latest_meta,
        "latest_event": event_rows[-1] if event_rows else None,
        "raw_image_url": raw_url,
        "processed_image_url": processed_url,
        "metadata_count": len(meta_rows),
        "event_count": len(event_rows),
    }


if __name__ == "__main__":
    print("Run server with:")
    print("uvicorn app:app --reload --host 0.0.0.0 --port 8000")