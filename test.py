import threading
from oureyes.puller import pull_stream
from models.fire_detection.fire_detection import fire_detection  # your existing fire_detection function

# --- Config ---
SRC_CAM = "fakefire"
FPS = 25
NUM_THREADS = 10

threads = []

for i in range(NUM_THREADS):
    frames = pull_stream(SRC_CAM)  # create a separate frame generator for each instance
    thread = threading.Thread(
        target=fire_detection,
        args=(frames, f"resultfakefire{i+1}", FPS),
        name=f"FireDetectionThread{i+1}"
    )
    threads.append(thread)
    thread.start()

# Wait for all threads to complete
for thread in threads:
    thread.join()
