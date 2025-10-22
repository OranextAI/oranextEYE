import threading
from oureyes.puller import pull_stream
from models.zone_analysis.zone_analysis import zone_analysis
from models.fire_detection.fire_detection import fire_detection

FPS = 20  # frames per second

def run_zone_analysis(frames, dest_cam):
    """Run zone analysis on provided frames to a specific output."""
    zone_analysis(frames, dest_cam=dest_cam, fps=FPS)

def run_fire_detection(frames):
    """Run fire detection on provided frames."""
    DEST_CAM_FIRE = "resultfakefire"
    fire_detection(frames, dest_cam=DEST_CAM_FIRE, fps=FPS)

if __name__ == "__main__":
    # ---------------- Config ----------------
    SRC_CAM_ZONE = "fakecam"
    SRC_CAM_FIRE = "fakefire"

    # ---------------- Pull Streams ----------------
    # Pull separate streams for each zone analysis thread
    zone_frames1 = pull_stream(SRC_CAM_ZONE)
    zone_frames2 = pull_stream(SRC_CAM_ZONE)
    fire_frames = pull_stream(SRC_CAM_FIRE)

    # ---------------- Threads ----------------
    # Zone analysis thread 1
    zone_thread1 = threading.Thread(
        target=run_zone_analysis,
        args=(zone_frames1, "resultfakecam1"),
        name="ZoneAnalysisThread1"
    )

    # Zone analysis thread 2
    zone_thread2 = threading.Thread(
        target=run_zone_analysis,
        args=(zone_frames2, "resultfakecam2"),
        name="ZoneAnalysisThread2"
    )

    # Fire detection thread
    fire_thread = threading.Thread(
        target=run_fire_detection,
        args=(fire_frames,),
        name="FireDetectionThread"
    )

    # ---------------- Run ----------------
    zone_thread1.start()
    zone_thread2.start()
    fire_thread.start()

    # Wait for threads to complete
    zone_thread1.join()
    zone_thread2.join()
    fire_thread.join()
