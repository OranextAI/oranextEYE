import threading
import cv2
import time
from oureyes.utils import build_rtsp_url
import numpy as np

# Shared containers
latest_frames = {}  # cam_name -> {"frame": np.array, "resolution": (w,h)}
locks = {}
threads = {}

# Config
RECONNECT_DELAY = 5
READ_SLEEP = 0.01
CAP_BUFFERSIZE = 1  # minimize internal buffer to keep latest frame

def _ensure_frame_shape(frame):
    """
    Ensure frame is in HxWx3 BGR format.
    Returns corrected frame or None if invalid.
    """
    if frame is None:
        return None

    # If grayscale -> convert to BGR
    if len(frame.shape) == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # If shape is (3, H, W) or (C, H, W) -> transpose to (H, W, C)
    if len(frame.shape) == 3 and frame.shape[0] in (1, 3) and frame.shape[0] < frame.shape[1]:
        # Heuristic: channels-first
        try:
            frame = np.transpose(frame, (1, 2, 0))
        except Exception:
            return None

    # If still not 3 channels, try to convert
    if len(frame.shape) != 3 or frame.shape[2] != 3:
        return None

    # Ensure width and height are integers and > 0
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return None

    return frame

def _reader_thread(cam_name):
    source = build_rtsp_url(cam_name)
    print(f"🎥 Starting reader for {cam_name}: {source}")
    cap = None

    while True:
        try:
            if cap is None or not cap.isOpened():
                print(f"🔄 Connecting to {source} ...")
                # Try using FFMPEG backend first (if OpenCV built with it)
                try:
                    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                except Exception:
                    cap = cv2.VideoCapture(source)
                # Try to set small buffer to reduce latency
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, CAP_BUFFERSIZE)
                except Exception:
                    pass

                if not cap.isOpened():
                    print(f"⚠️ Failed to open {source}. Retrying in {RECONNECT_DELAY}s.")
                    time.sleep(RECONNECT_DELAY)
                    continue
                print(f"✅ Connected to {source}")

            ret, frame = cap.read()
            if not ret:
                print(f"⚠️ Lost connection / read failed for {cam_name}. Reconnecting...")
                try:
                    cap.release()
                except Exception:
                    pass
                time.sleep(2)
                cap = None
                continue

            # Validate and fix shape if needed
            frame = _ensure_frame_shape(frame)
            if frame is None:
                print(f"⚠️ Received invalid frame from {cam_name}. Skipping frame.")
                time.sleep(READ_SLEEP)
                continue

            h, w = frame.shape[:2]

            # Store the latest frame and resolution together
            with locks[cam_name]:
                latest_frames[cam_name] = {"frame": frame, "resolution": (w, h)}

            time.sleep(READ_SLEEP)

        except Exception as e:
            print(f"❌ Exception in reader thread for {cam_name}: {e}")
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
            time.sleep(RECONNECT_DELAY)
            cap = None

def start_stream(cam_name):
    """Start the stream reader thread for a camera (idempotent)."""
    if cam_name not in threads:
        locks[cam_name] = threading.Lock()
        latest_frames[cam_name] = {"frame": None, "resolution": (0, 0)}
        t = threading.Thread(target=_reader_thread, args=(cam_name,), daemon=True)
        t.start()
        threads[cam_name] = t
        print(f"🚀 Stream started for {cam_name}")

def stop_stream(cam_name):
    """
    Soft stop: removes references. The thread is daemon and will exit on process end.
    (A more forceful stop would require thread signalling; keep simple for now.)
    """
    if cam_name in threads:
        print(f"🛑 stop_stream called for {cam_name} (thread is daemon; will stop on process exit)")
        threads.pop(cam_name, None)
        locks.pop(cam_name, None)
        latest_frames.pop(cam_name, None)

def get_latest_frame(cam_name):
    """
    Return a safe copy of the latest frame for cam_name or None.
    Also returns resolution as (w,h).
    """
    if cam_name not in latest_frames:
        return None, (0, 0)
    with locks[cam_name]:
        info = latest_frames[cam_name]
        frame = info.get("frame")
        res = info.get("resolution", (0, 0))
        return (frame.copy(), res) if frame is not None else (None, res)
