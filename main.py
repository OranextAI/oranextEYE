import threading
from oureyes.puller import pull_stream
from models.zone_analysis.zone_analysis import zone_analysis
from models.fire_detection.fire_detection import fire_detection

def run_zone_analysis(frames):
    """Run zone analysis on provided frames."""
    DEST_CAM_ZONE = "resultfakecam"
    zone_analysis(frames, dest_cam=DEST_CAM_ZONE, fps=FPS)

def run_fire_detection(frames):
    """Run fire detection on provided frames."""
    DEST_CAM_FIRE = "resultfakefire"
    fire_detection(frames, dest_cam=DEST_CAM_FIRE, fps=FPS)

if __name__ == "__main__":
    # ---------------- Config ----------------
    SRC_CAM_ZONE = "fakecam"
    SRC_CAM_FIRE = "fakefire"
    FPS = 20

    # ---------------- Pull Streams ----------------
    zone_frames = pull_stream(SRC_CAM_ZONE)
    fire_frames = pull_stream(SRC_CAM_FIRE)

    # ---------------- Threads ----------------
    zone_thread = threading.Thread(
        target=run_zone_analysis,
        args=(zone_frames,),
        name="ZoneAnalysisThread"
    )
    fire_thread = threading.Thread(
        target=run_fire_detection,
        args=(fire_frames,),
        name="FireDetectionThread"
    )

    # ---------------- Run ----------------
    zone_thread.start()
    fire_thread.start()

    # Wait for both threads to complete (they usually run indefinitely)
    zone_thread.join()
    fire_thread.join()
