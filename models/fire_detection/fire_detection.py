"""
fire_detection.py — Optimised version.

Changes vs original:
- Removed push_stream / FFmpeg encoding entirely.
- Uses emit_detections() to send JSON to the Angular overlay via Socket.IO.
- Model loaded via model_registry (loaded once, shared across all threads).
- Frame-skip logic preserved (run inference every FRAME_SKIP+1 frames).
"""

import cv2
import time
import torch
import numpy as np
from PIL import Image

from oureyes.emitter import emit_detections
from oureyes.model_registry import get_siglip, get_siglip_lock, DEVICE
from oureyes.notifier import notify_server

# ── Config ────────────────────────────────────────────────────────────────
MODEL_NAME = "prithivMLmods/Fire-Detection-Siglip2"
CONFIDENCE_THRESHOLD = 0.85
ALERT_INTERVAL = 1
FRAME_SKIP = 2   # GPU: run inference every 3rd frame (was 4)

_last_alert_time = 0


def fire_detection(frames, dest_cam: str, fps: int):
    """
    Consume frames, run fire/smoke classification, emit detections to Angular.

    Args:
        frames:   Generator yielding BGR numpy frames.
        dest_cam: Stream ID sent to Angular (e.g. "fakecam").
        fps:      Unused — kept for API compatibility.
    """
    global _last_alert_time

    device = torch.device(DEVICE)
    model, processor, model_enabled = get_siglip(MODEL_NAME)
    _infer_lock = get_siglip_lock(MODEL_NAME)

    # ── Grab first frame to get resolution ────────────────────────────────
    frame_iterator = iter(frames)
    try:
        first_frame = next(frame_iterator)
    except StopIteration:
        print("[fire_detection] No frames available")
        return

    W, H = first_frame.shape[1], first_frame.shape[0]
    print(f"[fire_detection] {dest_cam} — {W}x{H}, model_enabled={model_enabled}")

    last_predictions: dict = {}
    last_fire_detected = False

    def run_inference(frame) -> tuple[dict, bool]:
        """Run SigLIP on a single frame, return (predictions, fire_detected)."""
        nonlocal last_predictions, last_fire_detected
        if not model_enabled:
            return {}, False

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with _infer_lock:
            with torch.no_grad():
                logits = model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().tolist()

        labels = model.config.id2label
        predictions = {labels[i]: round(probs[i], 3) for i in range(len(probs))}
        fire_detected = any(
            lbl.lower() in ("fire", "smoke") and p >= CONFIDENCE_THRESHOLD
            for lbl, p in predictions.items()
        )
        last_predictions = predictions
        last_fire_detected = fire_detected
        return predictions, fire_detected

    def build_detections(predictions: dict, fire_detected: bool) -> list:
        """
        Always emit full probabilities + full-frame border.
        - No fire: green border + all class probabilities
        - Fire/smoke: red border + all class probabilities
        """
        result = []

        # Full-frame border — id="border" tells Angular to draw the frame outline
        # Use explicit "fire" keyword so Angular color logic is unambiguous
        border_label = "fire_detected" if fire_detected else "safe"
        result.append({
            "id": "border",
            "label": border_label,
            "conf": None,
            "box": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
        })

        # All class probabilities as text-only entries (zero-size box)
        for label, prob in predictions.items():
            result.append({
                "id": f"prob_{label}",
                "label": f"{label}: {prob*100:.1f}%",
                "conf": prob,
                "box": {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0},
            })

        return result

    def process(frame, frame_idx: int):
        global _last_alert_time
        run_inf = (frame_idx % (FRAME_SKIP + 1) == 0) or not last_predictions
        if run_inf:
            predictions, fire_detected = run_inference(frame)
        else:
            predictions, fire_detected = last_predictions, last_fire_detected

        detections = build_detections(predictions, fire_detected)
        emit_detections(dest_cam, detections, W, H)

        # Kafka alert
        if run_inf and fire_detected:
            for lbl, prob in predictions.items():
                if lbl.lower() in ("fire", "smoke") and prob >= CONFIDENCE_THRESHOLD:
                    if time.time() - _last_alert_time >= ALERT_INTERVAL:
                        notify_server("camera", {"data": "fire_detected",
                                                  "iddevice": 19, "idattribute": 6})
                        print(f"[fire_detection] 🔥 {lbl} {prob:.2f} — alert sent")
                        _last_alert_time = time.time()

    # ── Main loop ─────────────────────────────────────────────────────────
    process(first_frame, 0)
    for idx, frame in enumerate(frame_iterator, start=1):
        process(frame, idx)


if __name__ == "__main__":
    from oureyes.puller import pull_stream
    frames = pull_stream("cam2sub")
    fire_detection(frames, dest_cam="cam2sub", fps=25)
