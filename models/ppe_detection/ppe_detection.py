"""
ppe_detection.py — Optimised version.

Detects PPE compliance (Gloves, HairNet, Labcoat, Person) using YOLO.
Emits bounding-box detections to Angular overlay via Socket.IO.
Model loaded once via model_registry, shared across all threads.
"""

import cv2
import os

from oureyes.emitter import emit_detections
from oureyes.model_registry import get_yolo, get_yolo_lock

# ── Config ────────────────────────────────────────────────────────────────
CLASS_NAMES        = ['Gloves', 'HairNet', 'Labcoat', 'Person']
CONFIDENCE_THRESHOLD = 0.3
FRAME_SKIP         = 2   # run inference every 3rd frame on GPU


def ppe_detection(frames, dest_cam: str, fps: int):
    """
    Detect PPE and emit bounding-box detections to Angular.

    Args:
        frames:   Generator yielding BGR numpy frames.
        dest_cam: Socket.IO streamId (e.g. "ppe_detection_cam2sub").
        fps:      Unused — kept for API compatibility.
    """
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "best300.pt")

    frame_iterator = iter(frames)
    try:
        first_frame = next(frame_iterator)
    except StopIteration:
        print("[ppe_detection] No frames available")
        return

    W, H = first_frame.shape[1], first_frame.shape[0]
    print(f"[ppe_detection] {dest_cam} — {W}x{H}")

    model = get_yolo(MODEL_PATH)
    _infer_lock = get_yolo_lock(MODEL_PATH)

    last_detections: list = []

    def run_inference(frame) -> list:
        """Run YOLO and return normalised detection list."""
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with _infer_lock:
            results = model(
                img_rgb,
                conf=CONFIDENCE_THRESHOLD,
                classes=list(range(len(CLASS_NAMES))),
                verbose=False,
            )
        r = results[0] if isinstance(results, list) else results
        out=[]
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf   = float(box.conf[0])
            cls_id = int(box.cls[0])
            if conf >= CONFIDENCE_THRESHOLD and cls_id < len(CLASS_NAMES):
                out.append({
                    "label": CLASS_NAMES[cls_id],
                    "conf":  round(conf, 3),
                    "box": {
                        "x": round(x1 / W, 4),
                        "y": round(y1 / H, 4),
                        "w": round((x2 - x1) / W, 4),
                        "h": round((y2 - y1) / H, 4),
                    },
                })
        return out

    def process(frame, frame_idx: int):
        nonlocal last_detections
        run_inf = (frame_idx % (FRAME_SKIP + 1) == 0) or frame_idx == 0
        if run_inf:
            last_detections = run_inference(frame)

        # Always emit — if empty, send a "nothing detected" status entry
        detections_out = last_detections if last_detections else [{
            "id": "status",
            "label": "No PPE detected",
            "conf": None,
            "box": {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0},
        }]
        emit_detections(dest_cam, detections_out, W, H)

    process(first_frame, 0)
    for idx, frame in enumerate(frame_iterator, start=1):
        process(frame, idx)


if __name__ == "__main__":
    from oureyes.puller import pull_stream
    frames = pull_stream("cam2sub")
    ppe_detection(frames, dest_cam="cam2sub", fps=25)
