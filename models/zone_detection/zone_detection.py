import numpy as np
import supervision as sv
import cv2
import json
import os
from ultralytics import YOLO
from supervision.draw.color import Color, ColorPalette
from oureyes.pusher import push_stream

def zone_detection(frames, dest_cam, fps):
    """
    Process frames from a stream, detect objects in zones from JSON, annotate detections, and push to RTSP.

    Args:
        frames: Generator yielding frames (NumPy arrays) from pull_stream.
        dest_cam (str): Destination RTSP camera name (e.g., 'zone_detection').
        fps (int): Frames per second for the output stream.
    """
    # Configuration
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "startup.pt")
    ZONES_FILE = os.path.join(os.path.dirname(__file__), "zones_detection.json")
    CONFIDENCE_THRESHOLD = 0.1
    IMAGE_SIZE = 1280

    # Get frame resolution from the first frame
    frame_iterator = iter(frames)
    try:
        first_frame = next(frame_iterator)
    except StopIteration:
        print("Error: No frames available from input stream")
        return

    VIDEO_RESOLUTION = (first_frame.shape[1], first_frame.shape[0])
    print(f"Detected frame resolution: {VIDEO_RESOLUTION}")

    # Load zones from JSON
    if not os.path.exists(ZONES_FILE):
        print(f"Error: No zones file found at {ZONES_FILE}")
        return
    try:
        with open(ZONES_FILE, 'r') as f:
            zones_data = json.load(f)
    except Exception as e:
        print(f"Error loading zones file: {e}")
        return

    # Convert normalized zone coordinates to pixel coordinates
    polygons = []
    for zone in zones_data:
        poly = [[int(point['x'] * VIDEO_RESOLUTION[0]), int(point['y'] * VIDEO_RESOLUTION[1])] for point in zone]
        polygons.append(np.array(poly, dtype=np.int32))

    # Load YOLO model
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load model and move to device (YOLO supports .to() directly)
        model = YOLO(MODEL_PATH).to(device)
        print(f"Model '{MODEL_PATH}' loaded successfully on {device}.")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        import traceback
        traceback.print_exc()
        # Try loading without device specification as fallback
        try:
            model = YOLO(MODEL_PATH)
            print(f"Model '{MODEL_PATH}' loaded successfully (fallback, device auto-detected).")
        except Exception as e2:
            print(f"Fallback loading also failed: {e2}")
            return

    # Initialize annotator
    colors = ColorPalette(colors=[Color(255, 0, 0), Color(0, 255, 0), Color(0, 0, 255)])
    box_annotator = sv.BoxAnnotator(color=colors.by_idx(0), thickness=4)

    def process_frame(frame):
        """Process a single frame and return the annotated frame."""
        if (frame.shape[1], frame.shape[0]) != VIDEO_RESOLUTION:
            frame = cv2.resize(frame, VIDEO_RESOLUTION)

        # Perform object detection
        results = model(frame, imgsz=IMAGE_SIZE, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        if len(detections) == 0:
            # Draw polygons even if no detections
            for poly in polygons:
                cv2.polylines(frame, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
            return frame

        # Filter detections within the defined zones
        filtered_indices = []
        for i, (x1, y1, x2, y2) in enumerate(detections.xyxy):
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            for poly in polygons:
                if cv2.pointPolygonTest(poly, (center_x, center_y), False) >= 0:
                    filtered_indices.append(i)
                    break

        # Ensure filtered_indices is a numpy array of integers
        filtered_indices = np.array(filtered_indices, dtype=int)

        # Keep only the filtered detections
        detections_filtered = detections[filtered_indices]

        # Annotate the frame with filtered detections
        frame = box_annotator.annotate(scene=frame, detections=detections_filtered)

        # Draw the defined zones on the frame
        for poly in polygons:
            cv2.polylines(frame, [poly], isClosed=True, color=(0, 255, 0), thickness=2)

        return frame

    # Process frames and yield to push_stream
    def generate_processed_frames():
        yield process_frame(first_frame)
        for frame in frame_iterator:
            yield process_frame(frame)

    # Push processed frames to RTSP
    print("Processing stream...")
    push_stream(generate_processed_frames(), VIDEO_RESOLUTION[0], VIDEO_RESOLUTION[1], fps, dest_cam)
    print("Done.")

if __name__ == "__main__":
    from oureyes.puller import pull_stream
    SRC_CAM = "cam2sub"
    DEST_CAM = "fire2"
    FPS = 25
    frames = pull_stream(SRC_CAM)
    zone_detection(frames, dest_cam=DEST_CAM, fps=FPS)