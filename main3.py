import threading
from oureyes.puller import SharedPuller
from models.zone_analysis.zone_analysis import zone_analysis
from models.fire_detection.fire_detection import fire_detection
from oureyes.pusher import push_stream

FPS = 20

# --- Create shared puller (one per camera) ---
puller = SharedPuller("fakecam")

# --- Subscribe consumers ---
zone_queue = puller.subscribe()
fire_queue = puller.subscribe()

# --- Consumer threads ---
def run_zone_analysis():
    DEST_CAM_ZONE = "resultfakecam"
    while True:
        frame = zone_queue.get()
        zone_analysis([frame], dest_cam=DEST_CAM_ZONE, fps=FPS)

def run_fire_detection():
    DEST_CAM_FIRE = "resultfakefire"
    while True:
        frame = fire_queue.get()
        fire_detection([frame], dest_cam=DEST_CAM_FIRE, fps=FPS)

# --- Push stream example ---
def run_push():
    DEST_PUSH = "pushfakecam"
    width, height = 640, 480  # set your frame size
    frames = fire_queue  # or another queue
    while True:
        frame = frames.get()
        push_stream([frame], width, height, FPS, DEST_PUSH)

# --- Start threads ---
threading.Thread(target=run_zone_analysis, daemon=True).start()
threading.Thread(target=run_fire_detection, daemon=True).start()
threading.Thread(target=run_push, daemon=True).start()

# Keep main alive
while True:
    pass
