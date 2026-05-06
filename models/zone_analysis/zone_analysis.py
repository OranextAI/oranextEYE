"""
zone_analysis.py

Tracks worker (person) presence and time in zones using YOLO + ByteTrack.
Zones are stored in the DB (camera_zones table) — passed in via demo1.py.
Emits worker bounding boxes + zone status to Angular overlay via Socket.IO.
Model loaded once via model_registry, shared across all threads.

Fixes vs previous version:
- Inference runs on the full frame (not masked) — avoids missing partial detections
- PolygonZone objects created once, not per-frame
- Only persons inside a zone are emitted as bounding boxes
- imgsz fixed to 640 (standard YOLO size)
"""

import cv2
import os
import numpy as np
import supervision as sv

from oureyes.emitter import emit_detections
from oureyes.model_registry import get_yolo, get_yolo_lock

# ── Config ────────────────────────────────────────────────────────────────
PERSON_CLASS_ID         = 0       # COCO class 0 = person
CONFIDENCE_THRESHOLD    = 0.25
IMAGE_SIZE              = 640
BASE_MOVEMENT_THRESHOLD = 10
PATIENCE_FRAMES         = 15
FRAME_SKIP              = 2


def zone_analysis(frames, dest_cam: str, fps: int,
                  zone_points: list = None,
                  camera_id: int = None, camera_ai_id: int = None):
    """
    Track person presence in zones and emit detections to Angular.

    Args:
        frames:       Generator yielding BGR numpy frames.
        dest_cam:     Socket.IO streamId.
        fps:          Used for dt calculation.
        zone_points:  List of zone polygon lists [[{x,y},...], ...] normalised 0-1.
                      Passed in from demo1.py which reads them from the DB.
        camera_id:    Camera ID for event logging.
        camera_ai_id: CameraAI ID for event logging.
    """
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolov8n.pt")

    frame_iterator = iter(frames)
    try:
        first_frame = next(frame_iterator)
    except StopIteration:
        print("[zone_analysis] No frames available")
        return

    W, H = first_frame.shape[1], first_frame.shape[0]
    print(f"[zone_analysis] {dest_cam} — {W}x{H}")

    # ── Zones ─────────────────────────────────────────────────────────────
    if zone_points:
        zones_data = zone_points
    else:
        ZONES_FILE = os.path.join(os.path.dirname(__file__), "zones_perform.json")
        if os.path.exists(ZONES_FILE):
            import json
            print("[zone_analysis] Warning: using JSON fallback — run via demo1.py for DB zones")
            with open(ZONES_FILE) as f:
                zones_data = json.load(f)
        else:
            zones_data = []

    if not zones_data:
        print(f"[zone_analysis] No zones configured for {dest_cam} — emitting status only")

    # Pixel polygons for OpenCV point-in-polygon test
    polygons_px = [
        np.array([[int(p["x"] * W), int(p["y"] * H)] for p in zone], dtype=np.int32)
        for zone in zones_data
    ]

    # Create PolygonZone objects ONCE (not per frame)
    sv_zones = [sv.PolygonZone(polygon=p) for p in polygons_px]

    model       = get_yolo(MODEL_PATH)
    _infer_lock = get_yolo_lock(MODEL_PATH)
    tracker     = sv.ByteTrack()

    scale_factor       = W / 1280
    movement_threshold = BASE_MOVEMENT_THRESHOLD * scale_factor
    dt                 = 1.0 / max(fps, 1)

    worker_states: dict = {}
    workstation_data = [
        {
            "workers_present_now": 0,
            "unoccupied_time":     0.0,
            "total_presence_time": 0.0,
            "total_active_time":   0.0,
        }
        for _ in polygons_px
    ]

    last_detections: list = []

    def run_inference(frame) -> list:
        if (frame.shape[1], frame.shape[0]) != (W, H):
            frame = cv2.resize(frame, (W, H))

        # Reset per-frame counters
        for d in workstation_data:
            d["workers_present_now"] = 0

        # Run inference on the FULL frame — no masking
        with _infer_lock:
            results = model(
                frame, imgsz=IMAGE_SIZE, verbose=False,
                conf=CONFIDENCE_THRESHOLD, classes=[PERSON_CLASS_ID]
            )[0]

        sv_dets = sv.Detections.from_ultralytics(results)
        sv_dets = tracker.update_with_detections(sv_dets)

        # Age out stale workers
        for wid in list(worker_states.keys()):
            worker_states[wid]["frames_since_seen"] += 1
            if worker_states[wid]["frames_since_seen"] > PATIENCE_FRAMES:
                del worker_states[wid]

        out = []

        if sv_dets.tracker_id is not None:
            bottoms = sv_dets.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)

            for i, tid in enumerate(sv_dets.tracker_id):
                x1, y1, x2, y2 = sv_dets.xyxy[i]
                conf = float(sv_dets.confidence[i]) if sv_dets.confidence is not None else None
                pos  = bottoms[i]

                # Check which zone this person is in (use bottom-center point)
                cx, cy = float(pos[0]), float(pos[1])
                person_zone = -1
                for idx, poly in enumerate(polygons_px):
                    if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                        person_zone = idx
                        break

                if person_zone == -1:
                    # Person is outside all zones — skip
                    continue

                # Update zone stats
                workstation_data[person_zone]["workers_present_now"] += 1
                workstation_data[person_zone]["total_presence_time"] += dt

                moving = True
                if tid in worker_states and worker_states[tid]["station_id"] == person_zone:
                    dist = np.linalg.norm(
                        np.array(pos) - np.array(worker_states[tid]["last_pos"])
                    )
                    moving = dist > movement_threshold

                if moving:
                    workstation_data[person_zone]["total_active_time"] += dt

                worker_states[tid] = {
                    "station_id":       person_zone,
                    "last_pos":         pos,
                    "frames_since_seen": 0,
                }

                # Emit bounding box only for persons inside a zone
                out.append({
                    "id":    str(tid),
                    "label": "worker",
                    "conf":  round(conf, 3) if conf is not None else None,
                    "box": {
                        "x": round(x1 / W, 4),
                        "y": round(y1 / H, 4),
                        "w": round((x2 - x1) / W, 4),
                        "h": round((y2 - y1) / H, 4),
                    },
                })

        # Update unoccupied time for empty zones
        for d in workstation_data:
            if d["workers_present_now"] == 0:
                d["unoccupied_time"] += dt

        # Zone status labels
        for idx, data in enumerate(workstation_data):
            mins_absent,  secs_absent  = divmod(int(data["unoccupied_time"]),     60)
            mins_present, secs_present = divmod(int(data["total_presence_time"]), 60)

            if data["workers_present_now"] > 0:
                status = (
                    f"Zone {idx+1}: occupied ({data['workers_present_now']} worker"
                    f"{'s' if data['workers_present_now'] > 1 else ''}) | "
                    f"presence {mins_present:02d}m{secs_present:02d}s"
                )
            else:
                status = (
                    f"Zone {idx+1}: empty {mins_absent:02d}m{secs_absent:02d}s | "
                    f"total {mins_present:02d}m{secs_present:02d}s"
                )

            out.append({
                "id":    f"zone_status_{idx}",
                "label": status,
                "conf":  None,
                "box":   {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0},
            })

        if not out:
            out.append({
                "id":    "status",
                "label": "No workers in zones",
                "conf":  None,
                "box":   {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0},
            })

        return out

    def process(frame, frame_idx: int):
        nonlocal last_detections
        if (frame_idx % (FRAME_SKIP + 1) == 0) or frame_idx == 0:
            last_detections = run_inference(frame)
        emit_detections(dest_cam, last_detections, W, H)

    process(first_frame, 0)
    for idx, frame in enumerate(frame_iterator, start=1):
        process(frame, idx)


if __name__ == "__main__":
    from oureyes.puller import pull_stream
    frames = pull_stream("cam2sub")
    zone_analysis(frames, dest_cam="cam2sub", fps=25)
