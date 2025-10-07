import cv2
import os
from dotenv import load_dotenv, find_dotenv
import time

load_dotenv(find_dotenv())

def pull_stream():
    """
    Pull frames from VIDEO_PATH or RTSP_URL defined in .env.
    Automatically tries to reconnect if RTSP source fails.
    Returns a generator of frames.
    """
    source = os.getenv("RTSP_URL") or os.getenv("VIDEO_PATH")
    if not source:
        raise ValueError("RTSP_URL or VIDEO_PATH not defined in .env")

    while True:
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
                    print("⚠️ Frame read failed. Reconnecting...")
                    break
                yield frame
        finally:
            cap.release()
            time.sleep(1)  # short delay before reconnect
