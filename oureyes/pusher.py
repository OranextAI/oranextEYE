import subprocess
from oureyes.utils import build_rtsp_url

def push_stream(frames, width, height, fps, cam_name):
    """
    Push frames to RTSP destination built using cam_name.
    Frames are processed (YOLO + zones) before pushing.
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
        "-vf", f"scale={width}:{height},format=yuv420p",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-tune", "zerolatency",
        "-profile:v", "baseline",
        "-g", "30",
        "-pix_fmt", "yuv420p",
        "-f", "rtsp",
        dest_url
    ]

    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    try:
        for frame in frames:
            process.stdin.write(frame.tobytes())
    except Exception as e:
        print(f"⚠️ Error in push_stream: {e}")
    finally:
        process.stdin.close()
        process.wait()
