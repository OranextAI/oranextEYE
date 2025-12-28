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
CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold for fire/smoke

# Probability colors for each class (customize as needed)
PROB_COLORS = {
    "fire": (0, 0, 255),      # Red
    "smoke": (0, 165, 255),   # Orange
    "no_fire": (0, 255, 0)    # Green
}

def fire_detection(frames, dest_cam, fps):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = "prithivMLmods/Fire-Detection-Siglip2"
    try:
        model = SiglipForImageClassification.from_pretrained(model_name)
        processor = AutoImageProcessor.from_pretrained(model_name)
        model.to(device)
        print(f"✅ SigLIP model loaded from: {model_name} on {device}")
        print(f"📦 Model classes: {model.config.id2label}")
    except Exception as e:
        print(f"❌ Error loading SigLIP model: {e}")
        return

    BASE_FONT_SCALE = 1.2
    BASE_TEXT_THICKNESS = 2
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    TEXT_OFFSET_Y = 40

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

    def process_frame(frame):
        global _last_alert_time

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

        # Check for fire/smoke detection
        fire_detected = any(
            label.lower() in ["fire", "smoke"] and prob >= CONFIDENCE_THRESHOLD
            for label, prob in predictions.items()
        )

        # Draw border
        border_color = (0, 0, 255) if fire_detected else (0, 255, 0)  # Red if fire/smoke else green
        cv2.rectangle(frame, (0, 0), (VIDEO_RESOLUTION[0]-1, VIDEO_RESOLUTION[1]-1), border_color, BORDER_THICKNESS)

        # Draw probabilities
        for idx, (label, prob) in enumerate(predictions.items()):
            color = PROB_COLORS.get(label.lower(), (255, 255, 255))  # Default white
            text = f"{label}: {prob*100:.1f}%"
            y_pos = TEXT_OFFSET_Y * (idx + 1)
            # Draw text with outline for clarity
            cv2.putText(frame, text, (10, y_pos), FONT, FONT_SCALE, (0,0,0), TEXT_THICKNESS+2, lineType=cv2.LINE_AA)
            cv2.putText(frame, text, (10, y_pos), FONT, FONT_SCALE, color, TEXT_THICKNESS, lineType=cv2.LINE_AA)

        # Send alert if needed
        if fire_detected:
            for label, prob in predictions.items():
                if label.lower() in ["fire", "smoke"] and prob >= CONFIDENCE_THRESHOLD:
                    if time.time() - _last_alert_time >= ALERT_INTERVAL:
                        alert_data = {"data": f"fire_detected", "iddevice": 19, "idattribute": 6}
                        notify_server("camera", alert_data)
                        print(f"🔥 {label} detected (prob: {prob:.3f}) — alert sent!")
                        _last_alert_time = time.time()

        return frame

    def generate_processed_frames():
        yield process_frame(first_frame)
        for frame in frame_iterator:
            yield process_frame(frame)

    push_stream(generate_processed_frames(), VIDEO_RESOLUTION[0], VIDEO_RESOLUTION[1], fps, dest_cam)


if __name__ == "__main__":
    from oureyes.puller import pull_stream

    SRC_CAM = "cam2sub"
    DEST_CAM = "fire2"
    FPS = 25

    frames = pull_stream(SRC_CAM)
    fire_detection(frames, dest_cam=DEST_CAM, fps=FPS)
