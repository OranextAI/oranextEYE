import cv2
import time
from oureyes.puller import pull_stream
from oureyes.pusher import push_stream
from oureyes.notifier import notify_server  # Kafka notifier

# Keep track of last alert time to avoid spamming
_last_alert_time = 0
ALERT_INTERVAL = 10  # seconds between alerts


def process_frame(frame):
    """
    Process the frame and send a Kafka alert if conditions are met.
    """
    global _last_alert_time

    # --- Example AI detection placeholder ---
    fire_detected = True  # Replace this with your AI model inference

    # Send alert every ALERT_INTERVAL seconds
    if fire_detected and (time.time() - _last_alert_time >= ALERT_INTERVAL):
        alert_data = {
            "data": "fire_detected",
            "iddevice": 19,
            "idattribute": 6
        }
        notify_server("camera", alert_data)
        _last_alert_time = time.time()

    # You can also annotate the frame if needed
    # e.g., draw a rectangle or text on the frame

    return frame


def main():
    # 🔧 Camera configuration
    SRC_CAM = "fakecamera"
    DEST_CAM = "pulledcam"
    FPS = 25

    print(f"🎥 Source Camera: {SRC_CAM}")
    print(f"📡 Destination Camera: {DEST_CAM}")
    print(f"⏱️ FPS: {FPS}")

    # Pull frames from the source camera
    frames = pull_stream(SRC_CAM)

    # Get first frame to know dimensions
    first_frame = next(frames)
    height, width = first_frame.shape[:2]

    # Generator that processes each frame
    def processed_frames():
        yield process_frame(first_frame)
        for f in frames:
            yield process_frame(f)

    # Push processed frames to destination camera
    push_stream(processed_frames(), width, height, FPS, DEST_CAM)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
