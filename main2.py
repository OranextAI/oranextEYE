import cv2
import time
import threading
import queue
from oureyes.puller import pull_stream
from models.zone_analysis.zone_analysis import zone_analysis
from models.ppe_safety.ppe_detection import ppe_detection
from models.surveillance_zones.surveillance_zones import surveillance_zones
from models.time_count.time_count import time_count
from models.zone_detection.zone_detection import zone_detection
from models.fire_detection.fire_detection import fire_detection

# ---------------- CONFIG ----------------
FPS = 25
QUEUE_SIZE = 3  # Limit queue to prevent memory buildup

# ---------------- DISTRIBUTOR ----------------
def distribute_frames(frames, queues):
    """
    Read frames from a generator and put them into multiple queues.
    Each model gets its own copy of the frame.
    """
    for frame in frames:
        for q in queues:
            if not q.full():
                q.put(frame)

# ---------------- WORKER ----------------
def worker(model_func, q, dest_cam, fps, frame_skip=1):
    """
    Worker thread for a single model.
    frame_skip: process every Nth frame (1 = no skip)
    """
    def frame_iterator():
        frame_count = 0
        while True:
            frame = q.get()
            if frame is None:
                break
            frame_count += 1
            if frame_skip > 1 and frame_count % frame_skip != 0:
                continue  # skip frame for this model
            yield frame

    model_func(frame_iterator(), dest_cam, fps)

# ---------------- FAKECAM PROCESS ----------------
def run_fakecam_processes(frames, cam_name):
    """
    Run all models on one fakecam stream with separate outputs.
    Each model can optionally skip frames to reduce GPU load.
    """
    # Create one queue per model
    queues = [queue.Queue(maxsize=QUEUE_SIZE) for _ in range(5)]

    # Define models, suffix for output, and optional frame skip
    model_definitions = [
        (zone_analysis, "zone", 1),
        (ppe_detection, "ppe", 2),
        (surveillance_zones, "surveillance", 1),
        (time_count, "time", 3),
        (zone_detection, "zonedetect", 1)
    ]

    # Start worker thread for each model
    for q, (model_func, suffix, skip) in zip(queues, model_definitions):
        dest_cam = f"{cam_name}_{suffix}"  # e.g., fakecam_ppe
        threading.Thread(
            target=worker,
            args=(model_func, q, dest_cam, FPS, skip),
            name=f"{dest_cam}_Thread",
            daemon=True
        ).start()

    # Distribute frames to all queues
    distribute_frames(frames, queues)

# ---------------- MAIN ----------------
if __name__ == "__main__":
    # Input streams
    SRC_CAM_FAKECAM = "fakecam"
    SRC_CAM_FAKEFIRE = "fakefire"

    # Pull frames from cameras
    fakecam_frames = pull_stream(SRC_CAM_FAKECAM)
    fire_frames = pull_stream(SRC_CAM_FAKEFIRE)

    # Start fakecam processes (multiple models, shared frames)
    threading.Thread(
        target=run_fakecam_processes,
        args=(fakecam_frames, SRC_CAM_FAKECAM),
        name="FakecamProcessesThread",
        daemon=True
    ).start()

    # Start fire detection separately
    threading.Thread(
        target=fire_detection,
        args=(fire_frames, "result_fakefire", FPS),
        name="FireDetectionThread",
        daemon=True
    ).start()

    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping all threads...")
