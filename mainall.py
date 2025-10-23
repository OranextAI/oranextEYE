import threading
from oureyes.puller import pull_stream
from models.zone_analysis.zone_analysis import zone_analysis
from models.fire_detection.fire_detection import fire_detection
from models.ppe_detection.ppe_detection import ppe_detection
from models.surveillance_zones.surveillance_zones import surveillance_zones
from models.time_count.time_count import time_count
from models.zone_detection.zone_detection import zone_detection

FPS = 20  # frames per second

# ------------------ Run Functions ------------------
def run_zone_analysis(frames):
    zone_analysis(frames, dest_cam="resultfakezone", fps=FPS)

def run_fire_detection(frames):
    fire_detection(frames, dest_cam="resultfakefire", fps=FPS)

def run_ppe_detection(frames):
    ppe_detection(frames, dest_cam="resultfakeppe", fps=FPS)

def run_surveillance_zones(frames):
    surveillance_zones(frames, dest_cam="resultfakesurveillance", fps=FPS)

""" def run_time_count(frames):
    time_count(frames, dest_cam="resultfaketimecount", fps=FPS) """

def run_zone_detection(frames):
    zone_detection(frames, dest_cam="resultfakezonedetection", fps=FPS)


# ------------------ Main ------------------
if __name__ == "__main__":
    # ----------- Sources -----------
    SRC_CAM_ZONE = "fakecam"
    SRC_CAM_FIRE = "fakefire"
    SRC_CAM_PPE = "fakecam"
    SRC_CAM_SURV = "fakecam"
    SRC_CAM_ZONEDET = "fakecam"

    # ----------- Pull Streams -----------
    zone_frames = pull_stream(SRC_CAM_ZONE)
    fire_frames = pull_stream(SRC_CAM_FIRE)
    ppe_frames = pull_stream(SRC_CAM_PPE)
    surv_frames = pull_stream(SRC_CAM_SURV)
    zonedet_frames = pull_stream(SRC_CAM_ZONEDET)

    # ----------- Threads -----------
    threads = [
        threading.Thread(target=run_zone_analysis, args=(zone_frames,), name="ZoneAnalysisThread"),
        threading.Thread(target=run_fire_detection, args=(fire_frames,), name="FireDetectionThread"),
        threading.Thread(target=run_ppe_detection, args=(ppe_frames,), name="PPEDetectionThread"),
        threading.Thread(target=run_surveillance_zones, args=(surv_frames,), name="SurveillanceZonesThread"),
        threading.Thread(target=run_zone_detection, args=(zonedet_frames,), name="ZoneDetectionThread"),
    ]

    # ----------- Start Threads -----------
    for t in threads:
        t.start()

    # ----------- Wait for Completion -----------
    for t in threads:
        t.join()
