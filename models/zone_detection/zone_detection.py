"""
zone_detection.py — Optimised version.

Detects objects inside defined polygon zones using YOLO.
Emits zone polygons + bounding-box detections to Angular overlay via Socket.IO.
Model loaded once via model_registry, shared across all threads.
"""

import cv2
import json
import os
import numpy as np
import supervision as sv

from oureyes.emitter import emit_detections
from oureyes.model_registry import get_yolo, get_yolo_lock

# ── Config ────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.1
IMAGE_SIZE           = 1280
FRAME_SKIP           = 2


def zone_detection(frames, dest_cam: str, fps: int,
                   zone_points: list = None):
    """
    Detect objects inside defined zones and emit to Angular overlay.

    Args:
        frames:   Generator yielding BGR numpy frames.
        dest_cam: Socket.IO streamId.
        fps:      Unused — kept for API compatibility.
    """
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "startup.pt")
    ZONES_FILE = os.path.join(os.path.dirname(__file__), "zones_detection.json")

    frame_iterator = iter(frames)
    try:
        first_frame = next(frame_iterator)
    except StopIteration:
        print("[zone_detection] No frames available")
        return

    W, H = first_frame.shape[1], first_frame.shape[0]
    print(f"[zone_detection] {dest_cam} — {W}x{H}")

    if not os.path.exists(ZONES_FILE) and not zone_points:
        print(f"[zone_detection] No zones configured for {dest_cam}")
        def process_no_zones(frame, frame_idx):
            emit_detections(dest_cam, [{
                "id": "status", "label": "No zones configured",
                "conf": None, "box": {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0},
            }], W, H)
        process_no_zones(first_frame, 0)
        for idx, frame in enumerate(frame_iterator, start=1):
            process_no_zones(frame, idx)
        return

    # Zones come from DB via demo1.py (zone_points parameter)
    # JSON file is kept only as a standalone test fallback
    if zone_points:
        zones_data = zone_points
    elif os.path.exists(ZONES_FILE):
        print(f"[zone_detection] Warning: using JSON fallback — run via demo1.py for DB zones")
        with open(ZONES_FILE) as f:
            zones_data = json.load(f)
    else:
        zones_data = []

    # Pixel polygons for OpenCV point-in-polygon test
    polygons_px = [
        np.array([[int(p["x"] * W), int(p["y"] * H)] for p in zone], dtype=np.int32)
        for zone in zones_data
    ]

    # Normalised polygon points for Angular canvas
    zones_norm = [
        [{"x": p["x"], "y": p["y"]} for p in zone]
        for zone in zones_data
    ]

    model = get_yolo(MODEL_PATH)
    _infer_lock = get_yolo_lock(MODEL_PATH)
    last_detections: list = []

    def run_inference(frame) -> list:
        if (frame.shape[1], frame.shape[0]) != (W, H):
            frame = cv2.resize(frame, (W, H))

        with _infer_lock:
            results = model(frame, imgsz=IMAGE_SIZE, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        sv_dets  = sv.Detections.from_ultralytics(results)

        out = []

        # Object detections inside zones only (no polygon data — zones come from DB)
        in_zone_count = 0
        for i, (x1, y1, x2, y2) in enumerate(sv_dets.xyxy):
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            in_zone = any(
                cv2.pointPolygonTest(poly, (cx, cy), False) >= 0
                for poly in polygons_px
            )
            if not in_zone:
                continue
            in_zone_count += 1
            conf  = float(sv_dets.confidence[i]) if sv_dets.confidence is not None else None
            label = model.names[int(sv_dets.class_id[i])] if sv_dets.class_id is not None else "object"
            out.append({
                "label": label,
                "conf":  round(conf, 3) if conf is not None else None,
                "box": {
                    "x": round(x1 / W, 4), "y": round(y1 / H, 4),
                    "w": round((x2 - x1) / W, 4), "h": round((y2 - y1) / H, 4),
                },
            })

        if in_zone_count == 0:
            out.append({
                "id":    "status",
                "label": "No objects in zones",
                "conf":  None,
                "box":   {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0},
            })

        return out

    def process(frame, frame_idx: int):
        nonlocal last_detections
        run_inf = (frame_idx % (FRAME_SKIP + 1) == 0) or frame_idx == 0
        if run_inf:
            last_detections = run_inference(frame)
        emit_detections(dest_cam, last_detections, W, H)

    process(first_frame, 0)
    for idx, frame in enumerate(frame_iterator, start=1):
        process(frame, idx)


if __name__ == "__main__":
    from oureyes.puller import pull_stream
    frames = pull_stream("cam2sub")
    zone_detection(frames, dest_cam="cam2sub", fps=25)
