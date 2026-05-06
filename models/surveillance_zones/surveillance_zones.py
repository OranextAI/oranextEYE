"""
surveillance_zones.py — Optimised version.

Detects objects in surveillance zones, tracks absence timers per zone.
Zones come from DB via demo1.py (zone_points parameter).
Emits zone status + bounding boxes to Angular overlay via Socket.IO.
Model loaded once via model_registry, shared across all threads.
"""

import cv2
import os
import numpy as np
import supervision as sv

from oureyes.emitter import emit_detections
from oureyes.model_registry import get_yolo, get_yolo_lock

CONFIDENCE_THRESHOLD = 0.2
IMAGE_SIZE           = 1280
FRAME_SKIP           = 2


def surveillance_zones(frames, dest_cam: str, fps: int,
                       zone_points: list = None,
                       camera_id: int = None, camera_ai_id: int = None):
    """
    Monitor zones for object presence, track absence timers.

    Args:
        frames:       Generator yielding BGR numpy frames.
        dest_cam:     Socket.IO streamId.
        fps:          Used for absence timer calculation.
        zone_points:  List of zone polygon lists [[{x,y},...], ...] normalised 0-1.
                      Passed from demo1.py which reads from DB.
        camera_id:    Camera ID for event logging.
        camera_ai_id: CameraAI ID for event logging.
    """
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "startup.pt")

    frame_iterator = iter(frames)
    try:
        first_frame = next(frame_iterator)
    except StopIteration:
        print("[surveillance_zones] No frames available")
        return

    W, H = first_frame.shape[1], first_frame.shape[0]
    print(f"[surveillance_zones] {dest_cam} — {W}x{H}")

    # Zones from DB (preferred) or JSON fallback
    if zone_points:
        zones_data = zone_points
    else:
        ZONES_FILE = os.path.join(os.path.dirname(__file__), "zones_surveillance.json")
        if os.path.exists(ZONES_FILE):
            import json
            print(f"[surveillance_zones] Warning: using JSON fallback — run via demo1.py for DB zones")
            with open(ZONES_FILE) as f:
                zones_data = json.load(f)
        else:
            zones_data = []

    polygons_px = [
        np.array([[int(p["x"] * W), int(p["y"] * H)] for p in zone], dtype=np.int32)
        for zone in zones_data
    ]

    model = get_yolo(MODEL_PATH)
    _infer_lock = get_yolo_lock(MODEL_PATH)

    # Absence timer state per zone (frame number when zone became empty)
    zone_states = [{"absence_start_frame": None} for _ in polygons_px]
    frame_number = 0
    last_detections: list = []

    def run_inference(frame) -> list:
        nonlocal frame_number
        if (frame.shape[1], frame.shape[0]) != (W, H):
            frame = cv2.resize(frame, (W, H))

        with _infer_lock:
            results = model(frame, imgsz=IMAGE_SIZE, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        sv_dets  = sv.Detections.from_ultralytics(results)

        out = []

        for zone_idx, poly in enumerate(polygons_px):
            # Find detections inside this zone
            in_zone = []
            for i, (x1, y1, x2, y2) in enumerate(sv_dets.xyxy):
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0:
                    in_zone.append(i)

            count = len(in_zone)

            # Absence timer logic
            if count == 0:
                if zone_states[zone_idx]["absence_start_frame"] is None:
                    zone_states[zone_idx]["absence_start_frame"] = frame_number
                absence_frames = frame_number - zone_states[zone_idx]["absence_start_frame"]
                absence_secs   = absence_frames / fps
                mins, secs     = divmod(int(absence_secs), 60)
                if count == 0:
                    status = f"Zone {zone_idx+1}: empty {mins:02d}m{secs:02d}s"
            else:
                zone_states[zone_idx]["absence_start_frame"] = None
                if count == 1:
                    status = f"Zone {zone_idx+1}: OK (1 object)"
                else:
                    status = f"Zone {zone_idx+1}: check ({count} objects)"

            out.append({
                "id":    f"zone_status_{zone_idx}",
                "label": status,
                "conf":  None,
                "box":   {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0},
            })

            # Bounding boxes for detections inside this zone
            for i in in_zone:
                x1, y1, x2, y2 = sv_dets.xyxy[i]
                conf  = float(sv_dets.confidence[i]) if sv_dets.confidence is not None else None
                cls_id = int(sv_dets.class_id[i]) if sv_dets.class_id is not None else 0
                label = model.names.get(cls_id, str(cls_id))
                out.append({
                    "label": label,
                    "conf":  round(conf, 3) if conf is not None else None,
                    "box": {
                        "x": round(x1 / W, 4), "y": round(y1 / H, 4),
                        "w": round((x2 - x1) / W, 4), "h": round((y2 - y1) / H, 4),
                    },
                })

        if not out:
            out.append({
                "id": "status", "label": "No zones configured",
                "conf": None, "box": {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0},
            })

        frame_number += 1
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
    surveillance_zones(frames, dest_cam="cam2sub", fps=25)
