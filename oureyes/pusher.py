import os
import subprocess
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

def push_stream(frames, width, height, fps):
    """
    Push frames to MediaMTX RTSP destination defined in .env.
    """
    dest_url = os.getenv("DEST_URL")
    if not dest_url:
        host = os.getenv("HOST", "127.0.0.1")
        port = os.getenv("PORT", "8554")
        dest_url = f"rtsp://{host}:{port}/pulledcam"

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",  # stdin
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-tune", "zerolatency",
        "-f", "rtsp",
        dest_url
    ]

    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    try:
        for frame in frames:
            process.stdin.write(frame.tobytes())
    finally:
        process.stdin.close()
        process.wait()
