from oureyes.puller import pull_stream
from oureyes.pusher import push_stream
import os

def main():
    while True:
        frames = pull_stream()

        # Take first frame to get size
        try:
            first_frame = next(frames)
        except StopIteration:
            print("⚠️ No frames available. Retrying in 5s...")
            import time; time.sleep(5)
            continue

        height, width = first_frame.shape[:2]
        fps = int(os.getenv("FPS", 25))

        # Generator including first frame
        def frames_with_first():
            yield first_frame
            yield from frames

        try:
            push_stream(frames_with_first(), width, height, fps)
        except Exception as e:
            print("⚠️ Pusher failed:", e)
            import time; time.sleep(5)
            continue

if __name__ == "__main__":
    main()
