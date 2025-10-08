import subprocess
from oureyes.utils import build_rtsp_url


def push_stream(frames, width, height, fps, cam_name):
    """
    Push frames to RTSP destination built using cam_name.
    Example: cam_name="firecam" -> rtsp://20.199.8.131:8554/firecam
    """
    dest_url = build_rtsp_url(cam_name)
    print(f"🚀 Streaming to: {dest_url}")

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
