"""
zone_detection.py — Optimised version.

Detects objects inside defined polygon zones using YOLO.
Emits zone polygons + bounding-box detections to Angular overlay via Socket.IO.
Model loaded once via model_registry, shared across all threads.
"""

import cv2
import json
import os
import time
import numpy as np
import supervision as sv

from oureyes.emitter import emit_detections, emit_event
from oureyes.model_registry import get_yolo, get_yolo_lock

# ── Config ────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.1
IMAGE_SIZE           = 1280
FRAME_SKIP           = 2
ALERT_INTERVAL       = 10  # seconds between zone occupancy alerts


def zone_detection(frames, dest_cam: str, fps: int,
                   zone_points: list = None,
                   camera_id: int = None, camera_ai_id: int = None):
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
    _last_alert_times: dict = {}  # zone_index -> last alert timestamp

    def run_inference(frame) -> list:
        if (frame.shape[1], frame.shape[0]) != (W, H):
            frame = cv2.resize(frame, (W, H))

        with _infer_lock:
            results = model(frame, imgsz=IMAGE_SIZE, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        sv_dets = sv.Detections.from_ultralytics(results)

        # Count detections per zone
        zone_counts = [0] * len(polygons_px)
        for i, (x1, y1, x2, y2) in enumerate(sv_dets.xyxy):
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            for zi, poly in enumerate(polygons_px):
                if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                    zone_counts[zi] += 1

        total = len(sv_dets.xyxy)
        in_zone = sum(zone_counts)
        print(f"[zone_detection] {dest_cam} — {total} total detections, {in_zone} in zone")

        out = []
        # Emit zone polygons + occupancy status for Angular
        for z_idx, zone in enumerate(zones_norm):
            count = zone_counts[z_idx]
            if count > 0:
                label = f"occupied | {count} detected"
                # Fire zone occupancy event alert
                last_alert = _last_alert_times.get(z_idx, 0)
                if time.time() - last_alert >= ALERT_INTERVAL:
                    zone_label = zones_data[z_idx][0].get("label", f"Zone {z_idx+1}") if zones_data else f"Zone {z_idx+1}"
                    emit_event(
                        event_type="zone_occupied",
                        severity="warning",
                        camera_id=camera_id,
                        camera_ai_id=camera_ai_id,
                        message=f"{zone_label} occupied — {count} object(s) detected",
                        metadata={"zone_index": z_idx, "detection_count": count}
                    )
                    _last_alert_times[z_idx] = time.time()
            else:
                label = zones_data[z_idx][0].get("label", f"Zone {z_idx + 1}") if zones_data else f"Zone {z_idx + 1}"
            out.append({
                "id":    f"zone_status_{z_idx}",
                "label": label,
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
