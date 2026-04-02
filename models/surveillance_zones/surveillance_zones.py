"""
surveillance_zones.py — Optimised version.

Changes vs original:
- Removed push_stream / FFmpeg encoding entirely.
- Uses emit_detections() to send JSON to the Angular overlay via Socket.IO.
- Model loaded via model_registry (loaded once, shared across all threads).
- Zone absence timers preserved.
"""

import cv2
import json
import os
import numpy as np
import supervision as sv

from oureyes.emitter import emit_detections
from oureyes.model_registry import get_yolo

# ── Config ────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.2
IMAGE_SIZE = 1280


def surveillance_zones(frames, dest_cam: str, fps: int):
    """
    Detect objects in surveillance zones, track absence timers, emit to Angular.

    Args:
        frames:   Generator yielding BGR numpy frames.
        dest_cam: Stream ID sent to Angular.
        fps:      Used for absence timer calculation.
    """
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "startup.pt")
    ZONES_FILE = os.path.join(os.path.dirname(__file__), "zones_surveillance.json")

    frame_iterator = iter(frames)
    try:
        first_frame = next(frame_iterator)
    except StopIteration:
        print("[surveillance_zones] No frames available")
        return

    W, H = first_frame.shape[1], first_frame.shape[0]
    print(f"[surveillance_zones] {dest_cam} — {W}x{H}")

    if not os.path.exists(ZONES_FILE):
        print(f"[surveillance_zones] Zones file not found: {ZONES_FILE}")
        return
    with open(ZONES_FILE) as f:
        zones_data = json.load(f)

    polygons = [
        np.array([[int(p["x"] * W), int(p["y"] * H)] for p in zone], dtype=np.int32)
        for zone in zones_data
    ]

    model = get_yolo(MODEL_PATH)
    zone_states = [{"absence_start_frame": None} for _ in polygons]
    frame_number = 0

    def process(frame):
        nonlocal frame_number
        if (frame.shape[1], frame.shape[0]) != (W, H):
            frame = cv2.resize(frame, (W, H))

        results = model(frame, imgsz=IMAGE_SIZE, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        sv_dets = sv.Detections.from_ultralytics(results)

        detections_out = []

        for zone_idx, poly in enumerate(polygons):
            in_zone_mask = np.zeros(len(sv_dets), dtype=bool)
            for i, (x1, y1, x2, y2) in enumerate(sv_dets.xyxy):
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0:
                    in_zone_mask[i] = True

            count = int(np.sum(in_zone_mask))

            # Absence timer
            if count == 0:
                if zone_states[zone_idx]["absence_start_frame"] is None:
                    zone_states[zone_idx]["absence_start_frame"] = frame_number
                absence_frames = frame_number - zone_states[zone_idx]["absence_start_frame"]
                absence_secs = absence_frames / fps
                mins, secs = divmod(int(absence_secs), 60)
                zone_label = f"Zone {zone_idx+1}: empty — {mins:02d}:{secs:02d}"
            else:
                zone_states[zone_idx]["absence_start_frame"] = None
                zone_label = (f"Zone {zone_idx+1}: OK ({count})"
                              if count == 1 else f"Zone {zone_idx+1}: check ({count})")

            # Zone bounding box annotation
            bx, by, bw, bh = cv2.boundingRect(poly)
            detections_out.append({
                "id": f"zone_{zone_idx}",
                "label": zone_label,
                "conf": None,
                "box": {
                    "x": round(bx / W, 4), "y": round(by / H, 4),
                    "w": round(bw / W, 4), "h": round(bh / H, 4),
                },
            })

            # Individual detections inside this zone
            for i in np.where(in_zone_mask)[0]:
                x1, y1, x2, y2 = sv_dets.xyxy[i]
                conf = float(sv_dets.confidence[i]) if sv_dets.confidence is not None else None
                cls_id = int(sv_dets.class_id[i]) if sv_dets.class_id is not None else 0
                label = model.names.get(cls_id, str(cls_id))
                detections_out.append({
                    "label": label,
                    "conf": round(conf, 3) if conf is not None else None,
                    "box": {
                        "x": round(x1 / W, 4), "y": round(y1 / H, 4),
                        "w": round((x2 - x1) / W, 4), "h": round((y2 - y1) / H, 4),
                    },
                })

        emit_detections(dest_cam, detections_out, W, H)
        frame_number += 1

    process(first_frame)
    for frame in frame_iterator:
        process(frame)


if __name__ == "__main__":
    from oureyes.puller import pull_stream
    frames = pull_stream("cam2sub")
    surveillance_zones(frames, dest_cam="cam2sub", fps=25)
