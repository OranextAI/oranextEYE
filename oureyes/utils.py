import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

def build_webrtc_url(cam_name):
    """
    Build WEBRTC URL from camera name.
    """
    host = os.getenv("HOST_WEBRTC", "127.0.0.1")
    port = os.getenv("PORT_WEBRTC", "8889")
    return f"http://{host}:{port}/{cam_name}/whep"


def build_rtsp_url(cam_name: str) -> str:
    """
    Build RTSP URL from a camera name.
    Example: cam_name="fakecamera" -> rtsp://20.199.41.19:8554/fakecamera
    """
    host = os.getenv("HOST", "127.0.0.1")
    port = os.getenv("PORT", "8554")
    return f"rtsp://{host}:{port}/{cam_name}"

