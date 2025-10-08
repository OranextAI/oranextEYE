import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

def build_rtsp_url(cam_name):
    """
    Build RTSP URL from camera name.
    Example: cam_name="fakecamera" -> rtsp://20.199.8.131:8554/fakecamera
    """
    host = os.getenv("HOST", "127.0.0.1")
    port = os.getenv("PORT", "8554")
    return f"rtsp://{host}:{port}/{cam_name}"
