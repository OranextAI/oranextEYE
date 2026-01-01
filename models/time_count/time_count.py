import cv2
import numpy as np
import json
import os
from ultralytics import solutions
from oureyes.pusher import push_stream

def time_count(frames, dest_cam, fps):
    """
    Process frames from a stream, track objects in zones from JSON, label as Working/Not Working, and push to RTSP.

    Args:
        frames: Generator yielding frames (NumPy arrays) from pull_stream.
        dest_cam (str): Destination RTSP camera name (e.g., 'time_count').
        fps (int): Frames per second for the output stream.
    """
    # Configuration
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolov8n.pt")
    ZONES_FILE = os.path.join(os.path.dirname(__file__), "zones_time_count.json")

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
    region_polygons = []
    for zone in zones_data:
        poly = [[int(point['x'] * VIDEO_RESOLUTION[0]), int(point['y'] * VIDEO_RESOLUTION[1])] for point in zone]
        region_polygons.append(np.array(poly, dtype=np.int32))

    # Initialize TrackZone with the first region (if available)
    try:
        trackzone = solutions.TrackZone(
            show=False,  # Disable display for server environment
            region=region_polygons[0] if region_polygons else [],  # Use first region
            model=MODEL_PATH,
            verbose=False  # Suppress YOLO verbose output
        )
    except Exception as e:
        print(f"Error initializing TrackZone: {e}")
        return

    def is_inside_region(x, y, region):
        """Check if the point (x, y) is inside the polygon region."""
        return cv2.pointPolygonTest(region, (x, y), False) >= 0

    def process_frame(frame):
        """Process a single frame and return the annotated frame."""
        if (frame.shape[1], frame.shape[0]) != VIDEO_RESOLUTION:
            frame = cv2.resize(frame, VIDEO_RESOLUTION)

        # Perform object tracking
        results = trackzone.trackzone(frame)

        # If results is an image tensor, fall back to drawing polygons only
        if isinstance(results, np.ndarray):
            print("Warning: TrackZone returned an image array instead of detection results.")
            for poly in region_polygons:
                cv2.polylines(frame, [poly], isClosed=True, color=(255, 0, 0), thickness=2)
            return frame

        # Process bounding boxes if available
        if hasattr(results, 'boxes'):
            boxes = results.boxes.xyxy.cpu().numpy()
            ids = results.boxes.id.cpu().numpy() if hasattr(results.boxes, 'id') else None

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                obj_id = ids[i] if ids is not None else None
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # Check if the object is inside any region
                inside_any_region = False
                for poly in region_polygons:
                    if is_inside_region(center_x, center_y, poly):
                        inside_any_region = True
                        break

                # Label and color based on region status
                if inside_any_region:
                    box_color = (0, 255, 0)  # Green for Working
                    label = f"ID: {obj_id} - Working" if obj_id is not None else "Object - Working"
                else:
                    box_color = (0, 0, 255)  # Red for Not Working
                    label = f"ID: {obj_id} - Not Working" if obj_id is not None else "Object - Not Working"

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        # Draw all region polygons
        for poly in region_polygons:
            cv2.polylines(frame, [poly], isClosed=True, color=(255, 0, 0), thickness=2)

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
    time_count(frames, dest_cam=DEST_CAM, fps=FPS)