"""
time_count.py — Optimised version.

Changes vs original:
- Removed push_stream / FFmpeg encoding entirely.
- Uses emit_detections() to send JSON to the Angular overlay via Socket.IO.
- Model loaded via model_registry (loaded once, shared across all threads).
"""

import cv2
import json
import os
import numpy as np

from oureyes.emitter import emit_detections
from oureyes.model_registry import get_trackzone


def time_count(frames, dest_cam: str, fps: int):
    """
    Track objects in zones (working / not working) and emit detections to Angular.

    Args:
        frames:   Generator yielding BGR numpy frames.
        dest_cam: Stream ID sent to Angular.
        fps:      Unused — kept for API compatibility.
    """
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolov8n.pt")
    ZONES_FILE = os.path.join(os.path.dirname(__file__), "zones_time_count.json")

    frame_iterator = iter(frames)
    try:
        first_frame = next(frame_iterator)
    except StopIteration:
        print("[time_count] No frames available")
        return

    W, H = first_frame.shape[1], first_frame.shape[0]
    print(f"[time_count] {dest_cam} — {W}x{H}")

    if not os.path.exists(ZONES_FILE):
        print(f"[time_count] Zones file not found: {ZONES_FILE}")
        return
    with open(ZONES_FILE) as f:
        zones_data = json.load(f)

    region_polygons = [
        np.array([[int(p["x"] * W), int(p["y"] * H)] for p in zone], dtype=np.int32)
        for zone in zones_data
    ]

    first_region = region_polygons[0].tolist() if region_polygons else []
    trackzone = get_trackzone(MODEL_PATH, first_region)

    def in_any_region(cx, cy):
        return any(
            cv2.pointPolygonTest(poly, (cx, cy), False) >= 0
            for poly in region_polygons
        )

    def process(frame):
        if (frame.shape[1], frame.shape[0]) != (W, H):
            frame = cv2.resize(frame, (W, H))

        results = trackzone.trackzone(frame)
        detections_out = []

        if hasattr(results, "boxes") and results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            ids = (results.boxes.id.cpu().numpy()
                   if hasattr(results.boxes, "id") and results.boxes.id is not None
                   else None)

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                obj_id = str(int(ids[i])) if ids is not None else None
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                working = in_any_region(cx, cy)
                detections_out.append({
                    "id": obj_id,
                    "label": "Working" if working else "Not Working",
                    "conf": None,
                    "box": {
                        "x": round(x1 / W, 4), "y": round(y1 / H, 4),
                        "w": round((x2 - x1) / W, 4), "h": round((y2 - y1) / H, 4),
                    },
                })

        emit_detections(dest_cam, detections_out, W, H)

    process(first_frame)
    for frame in frame_iterator:
        process(frame)


if __name__ == "__main__":
    from oureyes.puller import pull_stream
    frames = pull_stream("cam2sub")
    time_count(frames, dest_cam="cam2sub", fps=25)
