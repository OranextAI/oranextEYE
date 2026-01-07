import cv2
import os
import numpy as np
import time
import torch
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
from oureyes.pusher import push_stream
from oureyes.notifier import notify_server

# Global variables
_last_alert_time = 0
ALERT_INTERVAL = 1  # Minimum time between alerts (in seconds)
CONFIDENCE_THRESHOLD = 0.65  # Confidence threshold for fire/smoke

# Probability colors for each class (customize as needed)
PROB_COLORS = {
    "fire": (0, 0, 255),      # Red
    "smoke": (0, 165, 255),   # Orange
    "no_fire": (0, 255, 0)    # Green
}

def fire_detection(frames, dest_cam, fps):
    # For stability with current library versions, force CPU execution.
    device = torch.device("cpu")
    print(f"Using device: {device}")

    model_name = "prithivMLmods/Fire-Detection-Siglip2"
    model = None
    processor = None
    model_enabled = True

    try:
        # Simple, robust load on CPU. We do not move the module with .to(),
        # which is where meta-tensor errors were occurring.
        model = SiglipForImageClassification.from_pretrained(model_name)
        processor = AutoImageProcessor.from_pretrained(model_name)
        model.eval()  # Set to evaluation mode
        print(f"✅ SigLIP model loaded from: {model_name} on CPU")
        print(f"📦 Model classes: {model.config.id2label}")

        # Run a small dummy inference once to detect meta-tensor issues
        # early and disable the model gracefully instead of crashing
        # the streaming pipeline.
        try:
            dummy_image = Image.new("RGB", (224, 224), 0)
            dummy_inputs = processor(images=dummy_image, return_tensors="pt")
            with torch.no_grad():
                _ = model(**dummy_inputs)
            print("✅ SigLIP dummy inference succeeded; model enabled.")
        except Exception as test_err:
            if "meta" in str(test_err).lower():
                print("⚠️ SigLIP meta-tensor issue detected; disabling fire model to keep stream stable.")
                model_enabled = False
            else:
                # Other unexpected issues should still surface
                raise

    except Exception as e:
        print(f"❌ Error loading SigLIP model: {e}")
        import traceback
        traceback.print_exc()
        # If we can't load the model at all, disable it but keep video flowing
        model_enabled = False

    BASE_FONT_SCALE = 1.2
    BASE_TEXT_THICKNESS = 2
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    TEXT_OFFSET_Y = 40

    # Run heavy inference only every N frames and reuse predictions in between
    FRAME_SKIP = 4  # process 1 out of (FRAME_SKIP + 1) frames

    frame_iterator = iter(frames)
    try:
        first_frame = next(frame_iterator)
    except StopIteration:
        print("❌ Error: No frames available from input stream")
        return

    VIDEO_RESOLUTION = (first_frame.shape[1], first_frame.shape[0])
    print(f"📏 Detected frame resolution: {VIDEO_RESOLUTION}")

    scale_factor = VIDEO_RESOLUTION[0] / 1280
    FONT_SCALE = BASE_FONT_SCALE * scale_factor
    TEXT_THICKNESS = max(1, int(BASE_TEXT_THICKNESS * scale_factor))
    BORDER_THICKNESS = max(4, int(5 * scale_factor))

    # Cache last predictions to reuse between inference frames
    last_predictions = None
    last_fire_detected = False

    def process_frame(frame, run_inference=True):
        """Process a frame.

        If run_inference is False, reuse last predictions to avoid running
        the transformer on every single frame, which keeps the stream smoother
        under high load or with many parallel models.
        If the model is disabled (e.g., due to meta-tensor issues), we
        skip inference entirely and just pass frames through with a
        neutral border.
        """
        nonlocal last_predictions, last_fire_detected
        global _last_alert_time

        # If model is disabled, skip inference and avoid any tensor ops
        if not model_enabled or model is None or processor is None:
            predictions = {}
            fire_detected = False
        elif run_inference or last_predictions is None:
            # Heavy path: run SigLIP on this frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().tolist()

            labels = model.config.id2label
            predictions = {labels[i]: round(probs[i], 3) for i in range(len(probs))}

            fire_detected = any(
                label.lower() in ["fire", "smoke"] and prob >= CONFIDENCE_THRESHOLD
                for label, prob in predictions.items()
            )

            # Update cache
            last_predictions = predictions
            last_fire_detected = fire_detected
        else:
            # Light path: reuse previously computed predictions
            predictions = last_predictions if last_predictions is not None else {}
            fire_detected = last_fire_detected

        # Draw border based on current fire status
        border_color = (0, 0, 255) if fire_detected else (0, 255, 0)  # Red if fire/smoke else green
        cv2.rectangle(frame, (0, 0), (VIDEO_RESOLUTION[0]-1, VIDEO_RESOLUTION[1]-1), border_color, BORDER_THICKNESS)

        # Draw probabilities from current (or cached) predictions
        for idx, (label, prob) in enumerate(predictions.items()):
            color = PROB_COLORS.get(label.lower(), (255, 255, 255))  # Default white
            text = f"{label}: {prob*100:.1f}%"
            y_pos = TEXT_OFFSET_Y * (idx + 1)
            # Draw text with outline for clarity
            cv2.putText(frame, text, (10, y_pos), FONT, FONT_SCALE, (0, 0, 0), TEXT_THICKNESS + 2, lineType=cv2.LINE_AA)
            cv2.putText(frame, text, (10, y_pos), FONT, FONT_SCALE, color, TEXT_THICKNESS, lineType=cv2.LINE_AA)

        # Send alert only when we actually ran inference on this frame
        if run_inference and fire_detected:
            for label, prob in predictions.items():
                if label.lower() in ["fire", "smoke"] and prob >= CONFIDENCE_THRESHOLD:
                    if time.time() - _last_alert_time >= ALERT_INTERVAL:
                        alert_data = {"data": f"fire_detected", "iddevice": 19, "idattribute": 6}
                        notify_server("camera", alert_data)
                        print(f"🔥 {label} detected (prob: {prob:.3f}) — alert sent!")
                        _last_alert_time = time.time()

        return frame

    def generate_processed_frames():
        # Always run inference on the very first frame to seed the cache
        frame_index = 0
        yield process_frame(first_frame, run_inference=True)
        frame_index += 1

        for frame in frame_iterator:
            # Run heavy inference only every (FRAME_SKIP + 1) frames
            run_inference = (frame_index % (FRAME_SKIP + 1) == 0)
            yield process_frame(frame, run_inference=run_inference)
            frame_index += 1

    push_stream(generate_processed_frames(), VIDEO_RESOLUTION[0], VIDEO_RESOLUTION[1], fps, dest_cam)


if __name__ == "__main__":
    from oureyes.puller import pull_stream

    SRC_CAM = "cam2sub"
    DEST_CAM = "fire2"
    FPS = 25

    frames = pull_stream(SRC_CAM)
    fire_detection(frames, dest_cam=DEST_CAM, fps=FPS)
