from oureyes.puller import pull_stream
from oureyes.pusher import push_stream
import cv2
import os

# Optional: import your model
# from .fire_model import detect_fire

def process_frame(frame):
    """
    Example frame processing for fire detection.
    Replace this with your real AI model inference.
    """
    # Here you could run your AI model
    # prediction = detect_fire(frame)
    # For demonstration, we just return the original frame
    return frame

def main():
    frames = pull_stream()

    # Take first frame to get size
    first_frame = next(frames)
    height, width = first_frame.shape[:2]
    fps = int(os.getenv("FPS", 25))

    # Generator including first frame + processing
    def processed_frames():
        yield process_frame(first_frame)
        for f in frames:
            yield process_frame(f)

    # Push to MediaMTX
    push_stream(processed_frames(), width, height, fps)

if __name__ == "__main__":
    main()
