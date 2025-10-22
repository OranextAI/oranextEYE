import cv2
from oureyes.puller import pull_stream
from oureyes.pusher import push_stream

def process_frame(frame):
    """
    Placeholder for frame processing.
    Currently returns the frame as-is.
    
    Args:
        frame (np.ndarray): Input frame (H x W x C)
    
    Returns:
        np.ndarray: Processed frame
    """
    # TODO: Add your processing logic here
    return frame

def process_stream(frames, dest_cam, fps):
    """
    Processes frames from a generator and pushes to an RTSP stream.
    Automatically detects the frame resolution from the first frame.
    
    Args:
        frames: Generator yielding frames
        dest_cam (str): Destination RTSP camera name
        fps (int): Output frames per second
    """
    frame_iterator = iter(frames)
    try:
        first_frame = next(frame_iterator)
    except StopIteration:
        print("Error: No frames available from input stream")
        return

    # Detect resolution from first frame
    height, width = first_frame.shape[:2]
    print(f"Detected resolution: {width}x{height}")

    def frame_generator():
        # Process first frame
        yield process_frame(first_frame)
        # Process remaining frames
        for frame in frame_iterator:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            yield process_frame(frame)

    # Push processed frames to RTSP
    push_stream(frame_generator(), width, height, fps, dest_cam)

if __name__ == "__main__":
    SRC_CAM = "fakefire"
    DEST_CAM = "resultfakefire"
    FPS = 25

    # Pull frames from the source
    frames = pull_stream(SRC_CAM)

    # Process and push frames
    process_stream(frames, DEST_CAM, FPS)
