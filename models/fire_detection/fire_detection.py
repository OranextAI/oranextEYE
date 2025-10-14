import cv2
import os
import numpy as np
import time
from ultralytics import YOLO
from oureyes.pusher import push_stream
from oureyes.notifier import notify_server

# Global variable to track the last alert time
_last_alert_time = 0
ALERT_INTERVAL = 1  # Minimum time between alerts (in seconds)

def fire_detection(frames, dest_cam, fps):
    """
    Process frames from a stream, detect fire using YOLO, send notifications, and push to RTSP.
    
    Args:
        frames: Generator yielding frames (NumPy arrays) from pull_stream.
        dest_cam (str): Destination RTSP camera name (e.g., 'processedcam').
        fps (int): Frames per second for the output stream.
    """
    # Constants
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "fire_detector.pt")
    BBOX_COLOR = (0, 255, 0)  # Green for bounding boxes
    TEXT_COLOR = (0, 255, 0)  # Green for text
    BASE_FONT_SCALE = 1.0  # Base font scale for 1280px width
    BASE_LINE_THICKNESS = 2  # Base line thickness for 1280px width
    BASE_TEXT_THICKNESS = 2  # Base text thickness for 1280px width
    FONT = cv2.FONT_HERSHEY_COMPLEX

    # Initialize YOLO model
    try:
        detector = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    # Get frame resolution from the first frame
    frame_iterator = iter(frames)
    try:
        first_frame = next(frame_iterator)
    except StopIteration:
        print("Error: No frames available from input stream")
        return

    # Determine resolution
    VIDEO_RESOLUTION = (first_frame.shape[1], first_frame.shape[0])  # (width, height)
    print(f"Detected frame resolution: {VIDEO_RESOLUTION}")

    # Scale parameters based on frame width (using 1280 as reference)
    scale_factor = VIDEO_RESOLUTION[0] / 1280
    FONT_SCALE = BASE_FONT_SCALE * scale_factor
    LINE_THICKNESS = max(1, int(BASE_LINE_THICKNESS * scale_factor))
    TEXT_THICKNESS = max(1, int(BASE_TEXT_THICKNESS * scale_factor))

    def process_frame(frame):
        """Process a single frame, annotate fire detections, send notifications, and return the annotated frame."""
        global _last_alert_time

        # Resize frame to target resolution (if needed)
        if (frame.shape[1], frame.shape[0]) != VIDEO_RESOLUTION:
            frame = cv2.resize(frame, VIDEO_RESOLUTION)

        # Run YOLO fire detection
        results = detector.predict(frame, imgsz=VIDEO_RESOLUTION[0], verbose=False, conf=0.60)

        # Check for fire detections and send notification if conditions are met
        fire_detected = False
        for result in results:
            if len(result.boxes) > 0:  # Any detections indicate fire
                fire_detected = True
                break

        if fire_detected and (time.time() - _last_alert_time >= ALERT_INTERVAL):
            alert_data = {
                "data": "fire_detected",
                "iddevice": 19,
                "idattribute": 6
            }
            notify_server("camera", alert_data)
            _last_alert_time = time.time()

        # Annotate frame with bounding boxes and labels
        for result in results:
            for bbox in result.boxes:
                x1, y1, x2, y2 = map(int, bbox.xyxy[0])
                class_id = int(bbox.cls)
                class_name = detector.names[class_id]

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), BBOX_COLOR, LINE_THICKNESS)

                # Draw label
                text_y = y1 + int(20 * scale_factor)
                cv2.putText(frame, class_name, (x1, text_y), FONT, FONT_SCALE, TEXT_COLOR, TEXT_THICKNESS)

        return frame

    # Process frames and yield to push_stream
    def generate_processed_frames():
        # Yield the first frame (already retrieved)
        yield process_frame(first_frame)
        # Process remaining frames
        for frame in frame_iterator:
            processed_frame = process_frame(frame)
            yield processed_frame

    # Push processed frames to RTSP
    push_stream(generate_processed_frames(), VIDEO_RESOLUTION[0], VIDEO_RESOLUTION[1], fps, dest_cam)

if __name__ == "__main__":
    from oureyes.puller import pull_stream
    SRC_CAM = "fakefire"
    DEST_CAM = "processedcam"
    FPS = 25

    # Pull frames from the source camera
    frames = pull_stream(SRC_CAM)

    # Run fire detection
    fire_detection(frames, dest_cam=DEST_CAM, fps=FPS)