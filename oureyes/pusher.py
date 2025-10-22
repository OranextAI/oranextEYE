import subprocess
from oureyes.utils import build_rtsp_url

def push_stream(frames, width, height, fps, cam_name):
    """
    Push frames to RTSP destination built using cam_name.
    Frames are processed (YOLO + zones) before pushing.
    Uses GPU (NVENC) for encoding to reduce CPU usage.
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
        "-i", "-",  # read frames from stdin
        "-vf", f"format=bgr0",  # NVENC prefers bgr0 or nv12
        "-c:v", "h264_nvenc",   # Use NVIDIA GPU encoder
        "-preset", "llhq",      # Low-latency, high-quality
        "-b:v", "5M",           # Bitrate (adjust as needed)
        "-g", "30",
        "-f", "rtsp",
        dest_url
    ]

    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    try:
        for frame in frames:
            # Convert frame to bgr0 (if needed) and send
            process.stdin.write(frame.tobytes())
    except Exception as e:
        print(f"⚠️ Error in push_stream: {e}")
    finally:
        process.stdin.close()
        process.wait()
