# ~/oranextEYE/models/safety/ppe_detection.py
import cv2
import os
import cvzone
import torch
import numpy as np
from ultralytics import YOLO
from oureyes.pusher import push_stream

def ppe_detection(frames, dest_cam, fps):
    """
    Process frames from a stream, detect PPE, and push to RTSP.

    Args:
        frames: Generator yielding frames (NumPy arrays) from pull_stream.
        dest_cam (str): Destination RTSP camera name (e.g., 'safety').
        fps (int): Frames per second for the output stream.
    """
    # Configuration
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "best300.pt")
    CONFIDENCE_THRESHOLD = 0.3
    CLASS_NAMES = ['Gloves', 'HairNet', 'Labcoat', 'Person']
    CLASS_COLORS = {
        'Gloves': (255, 200, 0),
        'HairNet': (0, 200, 255),
        'Labcoat': (255, 0, 150),
        'Person': (0, 255, 0)
    }
    TRANSPARENCY = 0.6
    FONT_SCALE = 1.2
    THICKNESS = 2
    OFFSET = 8

    # Get frame resolution from the first frame
    frame_iterator = iter(frames)
    try:
        first_frame = next(frame_iterator)
    except StopIteration:
        print("Error: No frames available from input stream")
        return

    VIDEO_RESOLUTION = (first_frame.shape[1], first_frame.shape[0])  # (width, height)
    print(f"Detected frame resolution: {VIDEO_RESOLUTION}")

    # Load YOLO model
    # Force CPU to avoid meta-tensor device transfer issues in some
    # torch/ultralytics combinations. This is more stable, at the
    # cost of running detections on CPU.
    device = "cpu"
    try:
        model = YOLO(MODEL_PATH)  # let Ultralytics handle device internally (CPU)
        print(f"[INFO] Model '{MODEL_PATH}' loaded successfully on {device}.")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    def draw_ppe_overlay(img, ppe_status, position, width, height):
        """Draw PPE status overlay on the frame."""
        overlay = img.copy()
        x, y = position
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (50, 50, 50), -1)
        img = cv2.addWeighted(overlay, TRANSPARENCY, img, 1 - TRANSPARENCY, 0)
        offset_y = y + 30
        for cls in CLASS_NAMES:
            status = ppe_status[cls]
            color = (0, 255, 0) if "ON" in status else (0, 0, 255)
            cvzone.putTextRect(img, f"{cls}: {status}", (x + 10, offset_y),
                               scale=FONT_SCALE, thickness=THICKNESS, colorB=(50, 50, 50),
                               colorT=(255, 255, 255), colorR=color, offset=OFFSET)
            offset_y += 30
        return img

    def process_frame(frame):
        """Process a single frame and return the annotated frame."""
        if (frame.shape[1], frame.shape[0]) != VIDEO_RESOLUTION:
            frame = cv2.resize(frame, VIDEO_RESOLUTION)

        # Convert to RGB for YOLO
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run detection with verbose=False to suppress YOLO output
        results = model(img_rgb, stream=True, conf=CONFIDENCE_THRESHOLD, classes=list(range(len(CLASS_NAMES))), verbose=False)

        detections = []
        glove_count = 0
        ppe_status = {cls: "MISSING" for cls in CLASS_NAMES}

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                if conf > CONFIDENCE_THRESHOLD and cls_id < len(CLASS_NAMES):
                    label = CLASS_NAMES[cls_id]
                    detections.append((cls_id, conf, (x1, y1, x2, y2)))
                    if label == "Gloves":
                        glove_count += 1

        # Sort detections (Person first)
        detections.sort(key=lambda x: (x[0] != 3, -x[1]))

        # Annotate detections
        for cls_id, conf, (x1, y1, x2, y2) in detections:
            label = CLASS_NAMES[cls_id]
            color = CLASS_COLORS.get(label, (255, 255, 255))
            ppe_status[label] = "ON"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, THICKNESS)
            cvzone.putTextRect(frame, f'{label} {conf:.2f}', (x1, max(35, y1)),
                               scale=FONT_SCALE, thickness=THICKNESS, colorB=color,
                               colorT=(255, 255, 255), colorR=color, offset=OFFSET)

        # Additional glove logic
        if glove_count == 1:
            ppe_status["Gloves"] = "1 Glove: ON"
        elif glove_count >= 2:
            ppe_status["Gloves"] = "2 Gloves: ON"

        # Draw overlay
        frame = draw_ppe_overlay(frame, ppe_status, (20, VIDEO_RESOLUTION[1] - 180), 250, 160)

        return frame

    # Process frames and yield to push_stream
    def generate_processed_frames():
        yield process_frame(first_frame)
        for frame in frame_iterator:
            yield process_frame(frame)

    # Push processed frames to RTSP
    push_stream(generate_processed_frames(), VIDEO_RESOLUTION[0], VIDEO_RESOLUTION[1], fps, dest_cam)

if __name__ == "__main__":
    from oureyes.puller import pull_stream
    SRC_CAM = "fakecam"
    DEST_CAM = "safety"
    FPS = 25
    frames = pull_stream(SRC_CAM)
    ppe_detection(frames, dest_cam=DEST_CAM, fps=FPS)