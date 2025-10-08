import cv2
import time
from oureyes.utils import build_rtsp_url

def pull_stream(cam_name):
    """
    Pull frames from RTSP stream built using cam_name.
    Example: cam_name="fakecamera" -> rtsp://20.199.8.131:8554/fakecamera
    """
    source = build_rtsp_url(cam_name)
    while True:
        print(f"🎥 Connecting to source: {source}")
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print(f"⚠️ Cannot open source: {source}. Retrying in 5s...")
            time.sleep(5)
            continue

        print(f"✅ Connected to source: {source}")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"⚠️ Frame read failed from {cam_name}. Reconnecting...")
                    break
                yield frame
        finally:
            cap.release()
            time.sleep(1)
