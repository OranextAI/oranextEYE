from oureyes.puller import pull_stream
from models.zone_analysis.zone_analysis import zone_analysis







if __name__ == "__main__":
    SRC_CAM = "fakecam"
    DEST_CAM = "processedcam"
    FPS = 25

    # Pull frames from the camera
    frames = pull_stream(SRC_CAM)

    # Process frames and push to RTSP
    zone_analysis(frames, dest_cam=DEST_CAM, fps=FPS)
