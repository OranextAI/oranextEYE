"""
time_count.py — Optimised version.

Tracks objects with IDs, labels them Working/Not Working based on zone presence.
Zones come from DB via demo1.py (zone_points parameter).
Emits bounding boxes with Working/Not Working labels to Angular overlay.
Model loaded once via model_registry, shared across all threads.
"""

import cv2
import os
import numpy as np

from oureyes.emitter import emit_detections
from oureyes.model_registry import get_yolo, get_yolo_lock

FRAME_SKIP = 2


def time_count(frames, dest_cam: str, fps: int,
               zone_points: list = None,
               camera_id: int = None, camera_ai_id: int = None):
    """
    Track objects and label Working/Not Working based on zone presence.

    Args:
        frames:       Generator yielding BGR numpy frames.
        dest_cam:     Socket.IO streamId.
        fps:          Unused — kept for API compatibility.
        zone_points:  List of zone polygon lists [[{x,y},...], ...] normalised 0-1.
        camera_id:    Camera ID for event logging.
        camera_ai_id: CameraAI ID for event logging.
    """
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolov8n.pt")

    frame_iterator = iter(frames)
    try:
        first_frame = next(frame_iterator)
    except StopIteration:
        print("[time_count] No frames available")
        return

    W, H = first_frame.shape[1], first_frame.shape[0]
    print(f"[time_count] {dest_cam} — {W}x{H}")

    # Zones from DB (preferred) or JSON fallback
    if zone_points:
        zones_data = zone_points
    else:
        ZONES_FILE = os.path.join(os.path.dirname(__file__), "zones_time_count.json")
        if os.path.exists(ZONES_FILE):
            import json
            print(f"[time_count] Warning: using JSON fallback — run via demo1.py for DB zones")
            with open(ZONES_FILE) as f:
                zones_data = json.load(f)
        else:
            zones_data = []

    if not zones_data:
        print(f"[time_count] No zones configured for {dest_cam}")

    region_polygons = [
        np.array([[int(p["x"] * W), int(p["y"] * H)] for p in zone], dtype=np.int32)
        for zone in zones_data
    ]

    # Use plain YOLO + supervision ByteTrack instead of TrackZone
    # (avoids TrackZone's init-time region binding which breaks model_registry caching)
    import supervision as sv
    model   = get_yolo(MODEL_PATH)
    _infer_lock = get_yolo_lock(MODEL_PATH)
    tracker = sv.ByteTrack()

    def in_any_region(cx: float, cy: float) -> bool:
        return any(
            cv2.pointPolygonTest(poly, (cx, cy), False) >= 0
            for poly in region_polygons
        )

    last_detections: list = []

    def run_inference(frame) -> list:
        if (frame.shape[1], frame.shape[0]) != (W, H):
            frame = cv2.resize(frame, (W, H))

        with _infer_lock:
            results = model(frame, verbose=False, conf=0.25)[0]
        sv_dets = sv.Detections.from_ultralytics(results)
        sv_dets = tracker.update_with_detections(sv_dets)

        out = []
        if sv_dets.tracker_id is not None:
            for i, tid in enumerate(sv_dets.tracker_id):
                x1, y1, x2, y2 = sv_dets.xyxy[i]
                cx, cy  = (x1 + x2) / 2, (y1 + y2) / 2
                working = in_any_region(cx, cy)
                out.append({
                    "id":    str(tid),
                    "label": "Working" if working else "Not Working",
                    "conf":  None,
                    "box": {
                        "x": round(x1 / W, 4), "y": round(y1 / H, 4),
                        "w": round((x2 - x1) / W, 4), "h": round((y2 - y1) / H, 4),
                    },
                })

        if not out:
            out.append({
                "id": "status", "label": "No objects detected",
                "conf": None, "box": {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0},
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
    time_count(frames, dest_cam="cam2sub", fps=25)
