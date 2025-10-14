import cv2
import json
import os
import numpy as np
import supervision as sv
from ultralytics import YOLO
from oureyes.pusher import push_stream

def zone_analysis(frames, dest_cam, fps):
    """
    Process frames from a stream, analyze zones for worker presence, and push to RTSP.
    
    Args:
        frames: Generator yielding frames (NumPy arrays) from pull_stream.
        dest_cam (str): Destination RTSP camera name (e.g., 'processedcam').
        fps (int): Frames per second for the output stream.
    """
    # Constants
    ZONES_FILE = os.path.join(os.path.dirname(__file__), "zones_perform.json")
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolov8n.pt")
    WORKER_CLASS_ID = 0
    BASE_MOVEMENT_THRESHOLD = 10  # Base threshold for 1280px width
    PATIENCE_FRAMES = 15
    ACTIVE_ZONE_COLOR = (0, 180, 0)
    INACTIVE_ZONE_COLOR = (0, 0, 180)
    ZONE_BORDER_COLOR = (255, 255, 255)
    TEXT_COLOR = (255, 255, 255)
    TRANSPARENCY = 0.3
    BASE_FONT_SCALE = 0.7  # Base font scale for 1280px width
    BASE_LINE_THICKNESS = 2  # Base line thickness for 1280px width
    BASE_TEXT_THICKNESS = 2  # Base text thickness for 1280px width
    FONT = cv2.FONT_HERSHEY_SIMPLEX

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
    MOVEMENT_THRESHOLD = BASE_MOVEMENT_THRESHOLD * scale_factor
    FONT_SCALE = BASE_FONT_SCALE * scale_factor
    LINE_THICKNESS = max(1, int(BASE_LINE_THICKNESS * scale_factor))
    TEXT_THICKNESS = max(1, int(BASE_TEXT_THICKNESS * scale_factor))

    # Load zones from file
    if not os.path.exists(ZONES_FILE):
        print(f"Error: No zones file found at {ZONES_FILE}")
        return
    with open(ZONES_FILE, 'r') as f:
        zones_data = json.load(f)

    # Convert normalized zone coordinates to pixel coordinates
    workstation_polygons = []
    for zone in zones_data:
        poly = [[int(point['x'] * VIDEO_RESOLUTION[0]), int(point['y'] * VIDEO_RESOLUTION[1])] for point in zone]
        workstation_polygons.append(np.array(poly, dtype=np.int32))

    # Initialize YOLO model
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    # Initialize tracker and zones
    tracker = sv.ByteTrack()
    workstation_zones = [sv.PolygonZone(polygon=poly) for poly in workstation_polygons]
    detection_mask = np.zeros(VIDEO_RESOLUTION[::-1], dtype=np.uint8)
    for poly in workstation_polygons:
        cv2.fillPoly(detection_mask, [poly], 255)

    # Initialize state tracking
    worker_states = {}
    workstation_data = [{'workers_present_now': 0, 'unoccupied_time': 0.0,
                        'total_presence_time': 0.0, 'total_active_time': 0.0} for _ in workstation_zones]
    dt = 1 / fps

    def draw_zone_info(frame, zone, data):
        """Draw zone information on the frame."""
        overlay = frame.copy()
        color = ACTIVE_ZONE_COLOR if data['workers_present_now'] > 0 else INACTIVE_ZONE_COLOR
        cv2.fillPoly(overlay, [zone.polygon], color)
        cv2.addWeighted(overlay, TRANSPARENCY, frame, 1 - TRANSPARENCY, 0, frame)
        cv2.polylines(frame, [zone.polygon], True, ZONE_BORDER_COLOR, LINE_THICKNESS)
        mins, secs = divmod(int(data['unoccupied_time']), 60)
        unoccupied_str = f"Unoccupied: {mins:02d}m{secs:02d}s"
        x, y, w, h = cv2.boundingRect(zone.polygon)
        (tw, th), _ = cv2.getTextSize(unoccupied_str, FONT, FONT_SCALE, TEXT_THICKNESS)
        # Adjust text position with scaled offset
        text_y = y + int(25 * scale_factor)
        cv2.putText(frame, unoccupied_str, (x + w // 2 - tw // 2, text_y), FONT, FONT_SCALE, TEXT_COLOR, TEXT_THICKNESS)

    def process_frame(frame):
        """Process a single frame and return the annotated frame."""
        # Resize frame to target resolution (if needed)
        if (frame.shape[1], frame.shape[0]) != VIDEO_RESOLUTION:
            frame = cv2.resize(frame, VIDEO_RESOLUTION)

        # Reset worker counts for this frame
        for d in workstation_data:
            d['workers_present_now'] = 0

        # Apply detection mask and run YOLO
        masked = cv2.bitwise_and(frame, frame, mask=detection_mask)
        results = model(masked, imgsz=VIDEO_RESOLUTION[0], verbose=False, conf=0.25, classes=[WORKER_CLASS_ID])[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        # Update worker states
        for wid in list(worker_states.keys()):
            worker_states[wid]['frames_since_seen'] += 1
            if worker_states[wid]['frames_since_seen'] > PATIENCE_FRAMES:
                del worker_states[wid]

        # Process detections
        if detections.tracker_id is not None:
            bottom_centers = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            for i, tid in enumerate(detections.tracker_id):
                pos = bottom_centers[i]
                for idx, zone in enumerate(workstation_zones):
                    if zone.trigger(detections=sv.Detections(xyxy=np.array([detections.xyxy[i]]))):
                        workstation_data[idx]['workers_present_now'] += 1
                        workstation_data[idx]['total_presence_time'] += dt
                        moving = True
                        if tid in worker_states and worker_states[tid]['station_id'] == idx:
                            dist = np.linalg.norm(np.array(pos) - np.array(worker_states[tid]['last_pos']))
                            moving = dist > MOVEMENT_THRESHOLD
                        if moving:
                            workstation_data[idx]['total_active_time'] += dt
                        worker_states[tid] = {'station_id': idx, 'last_pos': pos, 'frames_since_seen': 0}
                        break

        # Update unoccupied time
        for d in workstation_data:
            if d['workers_present_now'] == 0:
                d['unoccupied_time'] += dt

        # Draw zone information
        for idx, zone in enumerate(workstation_zones):
            draw_zone_info(frame, zone, workstation_data[idx])

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
    SRC_CAM = "fakecam"
    DEST_CAM = "processedcam"
    FPS = 25

    # Pull frames from the source camera
    frames = pull_stream(SRC_CAM)

    # Run zone analysis
    zone_analysis(frames, dest_cam=DEST_CAM, fps=FPS)