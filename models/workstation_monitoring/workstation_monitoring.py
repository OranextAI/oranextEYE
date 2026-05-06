"""
workstation_monitoring.py

Monitors worker presence at workstations using YOLO + ByteTrack.

Timers (all wall-clock, never frame-based):
  - absence_seconds     : current continuous absence episode (resets to 0 on return)
  - total_absent_secs   : cumulative sum of ALL absence chunks this session
  - total_presence_secs : cumulative time worker was present

Alerts (each fires once per trigger condition):
  - continuous_alert : absence_seconds >= absence_threshold
  - total_alert      : total_absent_secs >= total_absence_threshold

Zone colors emitted via label markers:
  - "occupied"  → green
  - "[ABSENT]"  → orange  (absent, continuous under threshold)
  - "[ALERT]"   → red     (continuous threshold exceeded)
  - "[TOTAL]"   → red     (total threshold exceeded)
"""

import cv2
import os
import time
import numpy as np
import supervision as sv

from oureyes.emitter import emit_detections, emit_event
from oureyes.model_registry import get_yolo, get_yolo_lock

# ── Config ────────────────────────────────────────────────────────────────
PERSON_CLASS_ID              = 0
CONFIDENCE_THRESHOLD         = 0.1
IMAGE_SIZE                   = 640
BASE_MOVEMENT_THRESHOLD      = 10
PATIENCE_FRAMES              = 15
FRAME_SKIP                   = 2
DEFAULT_ABSENCE_THRESHOLD       = 300    # seconds continuous
DEFAULT_TOTAL_ABSENCE_THRESHOLD = 3600   # seconds total


def workstation_monitoring(frames, dest_cam: str, fps: int,
                           zone_points: list = None,
                           zone_absence_thresholds: list = None,
                           zone_total_absence_thresholds: list = None,
                           camera_id: int = None, camera_ai_id: int = None):
    """
    Monitor workstation zones for worker presence/absence.

    Args:
        frames:                         Generator yielding BGR numpy frames.
        dest_cam:                       Socket.IO streamId.
        fps:                            Used for movement threshold scaling.
        zone_points:                    [[{x,y},...], ...] normalised 0-1.
        zone_absence_thresholds:        Continuous absence alert threshold per zone (seconds).
        zone_total_absence_thresholds:  Total cumulative absence alert threshold per zone (seconds).
        camera_id:                      Camera ID for event logging.
        camera_ai_id:                   CameraAI ID for event logging.
    """
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolov8n.pt")

    frame_iterator = iter(frames)
    try:
        first_frame = next(frame_iterator)
    except StopIteration:
        print("[workstation_monitoring] No frames available")
        return

    W, H = first_frame.shape[1], first_frame.shape[0]
    print(f"[workstation_monitoring] {dest_cam} — {W}x{H}")

    # ── Zones ─────────────────────────────────────────────────────────────
    if zone_points:
        zones_data = zone_points
    else:
        ZONES_FILE = os.path.join(os.path.dirname(__file__), "zones_workstation.json")
        if os.path.exists(ZONES_FILE):
            import json
            print("[workstation_monitoring] Warning: using JSON fallback")
            with open(ZONES_FILE) as f:
                zones_data = json.load(f)
        else:
            zones_data = []

    if not zones_data:
        print(f"[workstation_monitoring] No zones configured for {dest_cam}")

    polygons_px = [
        np.array([[int(p["x"] * W), int(p["y"] * H)] for p in zone], dtype=np.int32)
        for zone in zones_data
    ]
    n_zones = len(polygons_px)

    # Per-zone thresholds
    def _thresholds(src, default, n):
        if src and len(src) == n:
            return [int(t) for t in src]
        return [default] * n

    absence_thresholds       = _thresholds(zone_absence_thresholds,       DEFAULT_ABSENCE_THRESHOLD,       n_zones)
    total_absence_thresholds = _thresholds(zone_total_absence_thresholds, DEFAULT_TOTAL_ABSENCE_THRESHOLD, n_zones)

    print(f"[workstation_monitoring] {dest_cam} — {n_zones} zone(s)")
    for i in range(n_zones):
        print(f"  Zone {i+1}: continuous={absence_thresholds[i]}s  total={total_absence_thresholds[i]}s")

    model       = get_yolo(MODEL_PATH)
    _infer_lock = get_yolo_lock(MODEL_PATH)
    tracker     = sv.ByteTrack()

    scale_factor       = W / 1280
    movement_threshold = BASE_MOVEMENT_THRESHOLD * scale_factor

    worker_states: dict = {}

    # Per-zone state
    zone_data = [
        {
            "workers_present_now":    0,
            # Continuous absence episode
            "absence_start":          None,   # wall-clock start of current absence
            "absence_seconds":        0.0,    # current episode duration
            "continuous_alert_sent":  False,  # reset when worker returns
            # Total absence accumulator
            "total_absent_secs":      0.0,    # sum of all absence chunks
            "total_alert_sent":       False,  # never resets
            # Presence
            "presence_start":         None,
            "total_presence_secs":    0.0,
        }
        for _ in polygons_px
    ]

    last_detections: list = []

    def run_inference(frame) -> list:
        now = time.time()

        if (frame.shape[1], frame.shape[0]) != (W, H):
            frame = cv2.resize(frame, (W, H))

        for d in zone_data:
            d["workers_present_now"] = 0

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
                cx, cy = float(pos[0]), float(pos[1])

                person_zone = -1
                for zi, poly in enumerate(polygons_px):
                    if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                        person_zone = zi
                        break
                if person_zone == -1:
                    continue

                zone_data[person_zone]["workers_present_now"] += 1

                moving = True
                if tid in worker_states and worker_states[tid]["station_id"] == person_zone:
                    dist = np.linalg.norm(np.array(pos) - np.array(worker_states[tid]["last_pos"]))
                    moving = dist > movement_threshold

                worker_states[tid] = {"station_id": person_zone, "last_pos": pos, "frames_since_seen": 0}

                out.append({
                    "id":    str(tid),
                    "label": "worker",
                    "conf":  round(conf, 3) if conf is not None else None,
                    "box": {
                        "x": round(x1 / W, 4), "y": round(y1 / H, 4),
                        "w": round((x2 - x1) / W, 4), "h": round((y2 - y1) / H, 4),
                    },
                })

        # ── Update timers and fire alerts ─────────────────────────────────
        for idx, d in enumerate(zone_data):
            occupied = d["workers_present_now"] > 0

            if occupied:
                # ── Worker present ────────────────────────────────────────
                if d["absence_start"] is not None:
                    # Absence episode just ended — add chunk to total
                    chunk = now - d["absence_start"]
                    d["total_absent_secs"]     += chunk
                    d["absence_seconds"]        = 0.0
                    d["absence_start"]          = None
                    d["continuous_alert_sent"]  = False
                    print(f"[workstation_monitoring] Station {idx+1}: returned — chunk {chunk:.0f}s, total absent {d['total_absent_secs']:.0f}s")

                # Accumulate presence
                if d["presence_start"] is None:
                    d["presence_start"] = now
                d["total_presence_secs"] += now - d["presence_start"]
                d["presence_start"] = now

            else:
                # ── Zone empty ────────────────────────────────────────────
                d["presence_start"] = None

                if d["absence_start"] is None:
                    d["absence_start"] = now

                d["absence_seconds"] = now - d["absence_start"]

                # Continuous absence alert
                cont_thresh = absence_thresholds[idx]
                if not d["continuous_alert_sent"] and d["absence_seconds"] >= cont_thresh:
                    mins, secs = divmod(int(d["absence_seconds"]), 60)
                    emit_event(
                        event_type="workstation_absence",
                        severity="warning",
                        camera_id=camera_id,
                        camera_ai_id=camera_ai_id,
                        message=(
                            f"Station {idx+1} continuously absent for "
                            f"{mins:02d}m{secs:02d}s (threshold: {cont_thresh}s)"
                        ),
                        metadata={
                            "zone_index": idx, "type": "continuous",
                            "absence_seconds": round(d["absence_seconds"]),
                            "threshold_seconds": cont_thresh,
                        }
                    )
                    d["continuous_alert_sent"] = True
                    print(f"[workstation_monitoring] Station {idx+1}: continuous alert ({mins:02d}m{secs:02d}s)")

                # Total absence alert — check current total + ongoing chunk
                running_total = d["total_absent_secs"] + d["absence_seconds"]
                total_thresh  = total_absence_thresholds[idx]
                if not d["total_alert_sent"] and running_total >= total_thresh:
                    tmins, tsecs = divmod(int(running_total), 60)
                    emit_event(
                        event_type="workstation_total_absence",
                        severity="critical",
                        camera_id=camera_id,
                        camera_ai_id=camera_ai_id,
                        message=(
                            f"Station {idx+1} total absence reached "
                            f"{tmins:02d}m{tsecs:02d}s (threshold: {total_thresh}s)"
                        ),
                        metadata={
                            "zone_index": idx, "type": "total",
                            "total_absent_seconds": round(running_total),
                            "threshold_seconds": total_thresh,
                        }
                    )
                    d["total_alert_sent"] = True
                    print(f"[workstation_monitoring] Station {idx+1}: total absence alert ({tmins:02d}m{tsecs:02d}s)")

        # ── Build status labels — always one per zone ─────────────────────
        for idx, data in enumerate(zone_data):
            cont_secs    = data["absence_seconds"]
            total_absent = data["total_absent_secs"] + (cont_secs if data["absence_start"] else 0)
            pres_secs    = data["total_presence_secs"]

            ma, sa = divmod(int(cont_secs),    60)
            mt, st = divmod(int(total_absent), 60)
            mp, sp = divmod(int(pres_secs),    60)

            if data["workers_present_now"] > 0 or data["absence_start"] is None or cont_secs < 1.0:
                # Worker present, no absence started yet, or sub-second absence → green
                n = data["workers_present_now"]
                worker_str = f" ({n} worker{'s' if n > 1 else ''})" if n > 0 else ""
                status = (
                    f"Station {idx+1}: occupied{worker_str} | "
                    f"presence {mp:02d}m{sp:02d}s | total absent {mt:02d}m{st:02d}s"
                )
            else:
                if data["total_alert_sent"]:
                    marker = "[TOTAL]"
                elif data["continuous_alert_sent"]:
                    marker = "[ALERT]"
                else:
                    marker = "[ABSENT]"

                status = (
                    f"Station {idx+1}: {marker} {ma:02d}m{sa:02d}s | "
                    f"total absent {mt:02d}m{st:02d}s"
                )

            out.append({
                "id":    f"zone_status_{idx}",
                "label": status,
                "conf":  None,
                "box":   {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0},
            })

        # Only show generic status when no zones are configured at all
        if not polygons_px and not out:
            out.append({
                "id": "status", "label": "No workers detected",
                "conf": None, "box": {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0},
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
    workstation_monitoring(frames, dest_cam="cam2sub", fps=25)
