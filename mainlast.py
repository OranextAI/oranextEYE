import threading
import time
from oureyes.stream_manager import start_stream, get_latest_frame
from models.zone_analysis.zone_analysis import zone_analysis
from models.fire_detection.fire_detection import fire_detection

FPS = 20  # frames per second

# ---------------- Wrappers ----------------
def run_zone_analysis(src_cam, dest_cam):
    """Continuously feed latest frames to zone analysis."""
    print(f"🚀 Zone analysis started for {src_cam} -> {dest_cam}")
    while True:
        frame = get_latest_frame(src_cam)
        if frame is None:
            time.sleep(0.05)
            continue
        zone_analysis(frame, dest_cam=dest_cam, fps=FPS)

def run_fire_detection(src_cam):
    """Continuously feed latest frames to fire detection."""
    print(f"🚀 Fire detection started for {src_cam}")
    DEST_CAM_FIRE = "resultfakefire"
    while True:
        frame = get_latest_frame(src_cam)
        if frame is None:
            time.sleep(0.05)
            continue
        fire_detection(frame, dest_cam=DEST_CAM_FIRE, fps=FPS)

# ---------------- Main ----------------
if __name__ == "__main__":
    SRC_CAM_ZONE = "fakecam"
    SRC_CAM_FIRE = "fakefire"

    # Start one background reader per camera
    start_stream(SRC_CAM_ZONE)
    start_stream(SRC_CAM_FIRE)

    # Threads for analysis
    zone_thread1 = threading.Thread(
        target=run_zone_analysis,
        args=(SRC_CAM_ZONE, "resultfakecam1"),
        name="ZoneAnalysisThread1",
        daemon=True
    )

    zone_thread2 = threading.Thread(
        target=run_zone_analysis,
        args=(SRC_CAM_ZONE, "resultfakecam2"),
        name="ZoneAnalysisThread2",
        daemon=True
    )

    fire_thread = threading.Thread(
        target=run_fire_detection,
        args=(SRC_CAM_FIRE,),
        name="FireDetectionThread",
        daemon=True
    )

    # Start threads
    zone_thread1.start()
    zone_thread2.start()
    fire_thread.start()

    # Keep main alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("🛑 Stopping all threads...")
