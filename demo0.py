import asyncio
import threading
from oureyes.puller_bis import pull_stream
from models.fire_detection.fire_detection import fire_detection

FPS = 20  # frames per second


def sync_frame_generator(queue, loop):
    """
    Converts an asyncio.Queue to a synchronous blocking generator.
    """
    while True:
        frame = asyncio.run_coroutine_threadsafe(queue.get(), loop).result()
        yield frame


def run_fire_detection(queue, loop, dest_cam):
    fire_detection(
        sync_frame_generator(queue, loop),
        dest_cam=dest_cam,
        fps=FPS
    )


async def main_async():
    loop = asyncio.get_running_loop()

    # Pull streams
    cam1_queue = await pull_stream("cam1sub")
    cam2_queue = await pull_stream("cam2sub")

    threads = [
        threading.Thread(
            target=run_fire_detection,
            args=(cam1_queue, loop, "fire1"),
            name="FireDetectionCam1Thread",
            daemon=True
        ),
        threading.Thread(
            target=run_fire_detection,
            args=(cam2_queue, loop, "fire2"),
            name="FireDetectionCam2Thread",
            daemon=True
        ),
    ]

    for t in threads:
        t.start()

    # Keep event loop alive
    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main_async())
