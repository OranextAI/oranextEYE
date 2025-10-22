import numpy as np
import supervision as sv
import cv2
import json
import os
from ultralytics import YOLO
from supervision.draw.color import Color, ColorPalette
from oureyes.pusher import push_stream

def surveillance_zones(frames, dest_cam, fps):
    """
    Process frames from a stream, detect objects in zones from JSON, track absence timers, and push to RTSP.

    Args:
        frames: Generator yielding frames (NumPy arrays) from pull_stream.
        dest_cam (str): Destination RTSP camera name (e.g., 'surveillance_zones').
        fps (int): Frames per second for the output stream.
    """
    # Configuration
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "startup.pt")
    ZONES_FILE = os.path.join(os.path.dirname(__file__), "zones_surveillance.json")
    CONFIDENCE_THRESHOLD = 0.2
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
    with open(ZONES_FILE, 'r') as f:
        zones_data = json.load(f)

    # Convert normalized zone coordinates to pixel coordinates
    polygons = []
    for zone in zones_data:
        poly = [[int(point['x'] * VIDEO_RESOLUTION[0]), int(point['y'] * VIDEO_RESOLUTION[1])] for point in zone]
        polygons.append(np.array(poly, dtype=np.int32))

    # Load YOLO model
    try:
        model = YOLO(MODEL_PATH)
        print(f"Model '{MODEL_PATH}' loaded successfully.")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    # Initialize zone states
    zone_states = [{'absence_start_frame': None} for _ in polygons]
    color_palette = ColorPalette.from_hex(['#FF0000', '#00FF00', '#00FFFF', '#FFFF00', '#FF00FF', '#0000FF'])
    box_annotator = sv.BoxAnnotator(color=color_palette.by_idx(5), thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_color=Color.WHITE, color=color_palette.by_idx(5))

    def process_frame(frame, frame_number):
        """Process a single frame and return the annotated frame."""
        if (frame.shape[1], frame.shape[0]) != VIDEO_RESOLUTION:
            frame = cv2.resize(frame, VIDEO_RESOLUTION)

        # Run detection with verbose=False
        results = model(frame, imgsz=IMAGE_SIZE, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        annotated_frame = frame.copy()

        if not polygons:
            if len(detections) > 0:
                annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
            cv2.putText(annotated_frame, "No zones defined. Detecting all objects.", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            return annotated_frame

        all_detections_in_any_zone_mask = np.zeros(len(detections), dtype=bool)

        for zone_idx, poly in enumerate(polygons):
            zone_detections_mask = np.zeros(len(detections), dtype=bool)
            for i, (x1, y1, x2, y2) in enumerate(detections.xyxy):
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                if cv2.pointPolygonTest(poly, (float(center_x), float(center_y)), False) >= 0:
                    zone_detections_mask[i] = True
                    all_detections_in_any_zone_mask[i] = True

            count = np.sum(zone_detections_mask)
            absence_text = ""
            if count == 0:
                zone_color = (0, 0, 255)  # Red for empty
                zone_text = f"Zone {zone_idx+1}: Vide"
                if zone_states[zone_idx]['absence_start_frame'] is None:
                    zone_states[zone_idx]['absence_start_frame'] = frame_number
                else:
                    absence_duration_frames = frame_number - zone_states[zone_idx]['absence_start_frame']
                    absence_duration_seconds = absence_duration_frames / fps
                    minutes = int(absence_duration_seconds // 60)
                    seconds = int(absence_duration_seconds % 60)
                    absence_text = f"Absence: {minutes:02d}:{seconds:02d}"
            else:
                zone_states[zone_idx]['absence_start_frame'] = None
                if count == 1:
                    zone_color = (0, 255, 0)  # Green for OK
                    zone_text = f"Zone {zone_idx+1}: Travailleur au poste"
                else:
                    zone_color = (0, 255, 255)  # Yellow for alert
                    zone_text = f"Zone {zone_idx+1}: Verifier ({count})"

            cv2.polylines(annotated_frame, [poly], isClosed=True, color=zone_color, thickness=3)
            moments = cv2.moments(poly)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            else:
                cx, cy = poly[0][0], poly[0][1]

            text_size = cv2.getTextSize(zone_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = cx - text_size[0] // 2
            text_y = cy + text_size[1] // 2

            cv2.putText(annotated_frame, zone_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(annotated_frame, zone_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

            if absence_text:
                absence_text_size = cv2.getTextSize(absence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                absence_text_x = cx - absence_text_size[0] // 2
                absence_text_y = text_y + text_size[1] + 10
                cv2.putText(annotated_frame, absence_text, (absence_text_x, absence_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(annotated_frame, absence_text, (absence_text_x, absence_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        detections_to_annotate = detections[all_detections_in_any_zone_mask]
        if len(detections_to_annotate) > 0:
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections_to_annotate)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections_to_annotate)

        return annotated_frame

    # Process frames and yield to push_stream
    def generate_processed_frames():
        frame_count = 0
        yield process_frame(first_frame, frame_count)
        for frame in frame_iterator:
            frame_count += 1
            yield process_frame(frame, frame_count)

    # Push processed frames to RTSP
    print("Processing stream...")
    push_stream(generate_processed_frames(), VIDEO_RESOLUTION[0], VIDEO_RESOLUTION[1], fps, dest_cam)
    print("Done.")

if __name__ == "__main__":
    from oureyes.puller import pull_stream
    SRC_CAM = "fakecam"
    DEST_CAM = "zones"
    FPS = 25
    frames = pull_stream(SRC_CAM)
    surveillance_zones(frames, dest_cam=DEST_CAM, fps=FPS)