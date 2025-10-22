import time
from oureyes.stream_manager import start_stream, get_latest_frame

if __name__ == "__main__":
    cams = ["fakecam", "fakefire"]  # use your actual cam names
    for c in cams:
        start_stream(c)

    try:
        while True:
            for c in cams:
                frame, res = get_latest_frame(c)
                if frame is None:
                    print(f"[{c}] no frame yet")
                else:
                    print(f"[{c}] frame shape: {frame.shape}  resolution stored: {res}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping test.")
