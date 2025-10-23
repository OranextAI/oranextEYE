import subprocess
import sys
from oureyes.utils import build_rtsp_url

def push_stream(frames, width, height, fps, cam_name):
    """
    Push frames to RTSP using NVENC and TCP transport.
    All ffmpeg output is silenced.
    """
    dest_url = build_rtsp_url(cam_name)
    print(f"🚀 Streaming to: {dest_url}")

    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner",            # hide ffmpeg banner
        "-loglevel", "error",      # show only errors
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",                 # read frames from stdin
        "-vf", "format=bgr0",      # NVENC prefers bgr0 or nv12
        "-c:v", "h264_nvenc",
        "-preset", "p4",           # good performance preset
        "-tune", "ll",             # low latency
        "-b:v", "5M",
        "-g", "30",
        "-f", "rtsp",
        "-rtsp_transport", "tcp",  # force TCP transport
        dest_url
    ]

    # Run ffmpeg silently (no stdout/stderr)
    process = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    try:
        for frame in frames:
            process.stdin.write(frame.tobytes())
    except KeyboardInterrupt:
        print("🛑 Streaming interrupted by user")
    except Exception as e:
        print(f"⚠️ Error in push_stream: {e}")
    finally:
        try:
            process.stdin.close()
        except Exception:
            pass
        process.wait()
        print("✅ Stream stopped")
