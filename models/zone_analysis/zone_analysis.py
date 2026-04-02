"""
zone_analysis.py — Optimised version.

Tracks worker presence and time in zones using YOLO + ByteTrack.
Zones are stored in the DB (camera_zones table) — not read from JSON.
Emits worker bounding boxes + zone status to Angular overlay via Socket.IO.
Model loaded once via model_registry, shared across all threads.
"""

import cv2
import os
import numpy as np
import supervision as sv

from oureyes.emitter import emit_detections
from oureyes.model_registry import get_yolo

# ── Config ────────────────────────────────────────────────────────────────
WORKER_CLASS_ID      = 0
BASE_MOVEMENT_THRESHOLD = 10
PATIENCE_FRAMES      = 15
FRAME_SKIP           = 2


def zone_analysis(frames, dest_cam: str, fps: int,
                  zone_points: list = None):
    """
    Track worker presence in zones and emit detections to Angular.

    Args:
        frames:      Generator yielding BGR numpy frames.
        dest_cam:    Socket.IO streamId.
        fps:         Used for dt calculation.
        zone_points: List of zone polygon lists [[{x,y},...], ...] normalised 0-1.
                     Passed in from demo1.py which reads them from the DB.
                     Falls back to zones_perform.json if not provided.
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

    # Zones come from DB via demo1.py (zone_points parameter)
    # JSON file is kept only as a standalone test fallback
    if zone_points:
        zones_data = zone_points
    else:
        ZONES_FILE = os.path.join(os.path.dirname(__file__), "zones_perform.json")
        if os.path.exists(ZONES_FILE):
            import json
            print(f"[zone_analysis] Warning: using JSON fallback — run via demo1.py for DB zones")
            with open(ZONES_FILE) as f:
                zones_data = json.load(f)
        else:
            zones_data = []

    if not zones_data:
        print(f"[zone_analysis] No zones — emitting status only")

    polygons_px = [
        np.array([[int(p["x"] * W), int(p["y"] * H)] for p in zone], dtype=np.int32)
        for zone in zones_data
    ]

    model   = get_yolo(MODEL_PATH)
    tracker = sv.ByteTrack()

    detection_mask = np.zeros((H, W), dtype=np.uint8)
    for poly in polygons_px:
        cv2.fillPoly(detection_mask, [poly], 255)

    scale_factor       = W / 1280
    movement_threshold = BASE_MOVEMENT_THRESHOLD * scale_factor
    dt                 = 1.0 / fps

    worker_states: dict = {}
    workstation_data = [
        {"workers_present_now": 0, "unoccupied_time": 0.0,
         "total_presence_time": 0.0, "total_active_time": 0.0}
        for _ in polygons_px
    ]

    last_detections: list = []

    def run_inference(frame) -> list:
        if (frame.shape[1], frame.shape[0]) != (W, H):
            frame = cv2.resize(frame, (W, H))

        for d in workstation_data:
            d["workers_present_now"] = 0

        masked  = cv2.bitwise_and(frame, frame, mask=detection_mask)
        results = model(masked, imgsz=W, verbose=False, conf=0.25,
                        classes=[WORKER_CLASS_ID])[0]
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
                pos  = bottoms[i]
                x1, y1, x2, y2 = sv_dets.xyxy[i]
                conf = float(sv_dets.confidence[i]) if sv_dets.confidence is not None else None

                for idx, zone in enumerate(
                    [sv.PolygonZone(polygon=p) for p in polygons_px]
                ):
                    if zone.trigger(detections=sv.Detections(
                            xyxy=np.array([sv_dets.xyxy[i]]))):
                        workstation_data[idx]["workers_present_now"] += 1
                        workstation_data[idx]["total_presence_time"] += dt
                        moving = True
                        if tid in worker_states and worker_states[tid]["station_id"] == idx:
                            dist = np.linalg.norm(
                                np.array(pos) - np.array(worker_states[tid]["last_pos"]))
                            moving = dist > movement_threshold
                        if moving:
                            workstation_data[idx]["total_active_time"] += dt
                        worker_states[tid] = {
                            "station_id": idx, "last_pos": pos, "frames_since_seen": 0}
                        break

                out.append({
                    "id":    str(tid),
                    "label": "worker",
                    "conf":  round(conf, 3) if conf is not None else None,
                    "box": {
                        "x": round(x1 / W, 4), "y": round(y1 / H, 4),
                        "w": round((x2 - x1) / W, 4), "h": round((y2 - y1) / H, 4),
                    },
                })

        # Update unoccupied time
        for d in workstation_data:
            if d["workers_present_now"] == 0:
                d["unoccupied_time"] += dt

        # Zone status annotations — rich counter info per zone
        for idx, data in enumerate(workstation_data):
            mins_absent,  secs_absent  = divmod(int(data["unoccupied_time"]),    60)
            mins_present, secs_present = divmod(int(data["total_presence_time"]), 60)

            if data["workers_present_now"] > 0:
                status = f"Zone {idx+1}: occupied | presence {mins_present:02d}m{secs_present:02d}s"
            else:
                status = f"Zone {idx+1}: empty {mins_absent:02d}m{secs_absent:02d}s | total {mins_present:02d}m{secs_present:02d}s"

            out.append({
                "id":    f"zone_status_{idx}",
                "label": status,
                "conf":  None,
                "box":   {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0},
            })

        if not out:
            out.append({
                "id": "status", "label": "No workers detected",
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
    zone_analysis(frames, dest_cam="cam2sub", fps=25)
