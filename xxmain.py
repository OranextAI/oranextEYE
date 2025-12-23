import asyncio
import threading
from oureyes.puller_bis import pull_stream

from models.zone_analysis.zone_analysis import zone_analysis
from models.fire_detection.fire_detection import fire_detection
from models.ppe_detection.ppe_detection import ppe_detection
from models.surveillance_zones.surveillance_zones import surveillance_zones
from models.zone_detection.zone_detection import zone_detection

FPS = 20  # frames per second


def sync_frame_generator(queue, loop):
    """
    Converts an asyncio.Queue to a synchronous blocking generator.
    """
    while True:
        frame = asyncio.run_coroutine_threadsafe(queue.get(), loop).result()
        yield frame


def run_zone_analysis(queue, loop):
    zone_analysis(sync_frame_generator(queue, loop), dest_cam="resultfakezone", fps=FPS)


def run_fire_detection(queue, loop):
    fire_detection(sync_frame_generator(queue, loop), dest_cam="resultfakefire", fps=FPS)


def run_ppe_detection(queue, loop):
    ppe_detection(sync_frame_generator(queue, loop), dest_cam="fakeppe", fps=FPS)


def run_surveillance_zones(queue, loop):
    surveillance_zones(sync_frame_generator(queue, loop), dest_cam="resultfakesurveillance", fps=FPS)


def run_zone_detection(queue, loop):
    zone_detection(sync_frame_generator(queue, loop), dest_cam="resultfakezonedetection", fps=FPS)


async def main_async():
    loop = asyncio.get_running_loop()

    # Await queues from async pull_stream
    zone_queue = await pull_stream("cam1sub")
    fire_queue = await pull_stream("cam1sub")
    ppe_queue = await pull_stream("cam1sub")
    surv_queue = await pull_stream("cam1sub")
    zonedet_queue = await pull_stream("cam1sub")

    threads = [
        threading.Thread(target=run_zone_analysis, args=(zone_queue, loop), name="ZoneAnalysisThread"),
        threading.Thread(target=run_fire_detection, args=(fire_queue, loop), name="FireDetectionThread"),
        threading.Thread(target=run_ppe_detection, args=(ppe_queue, loop), name="PPEDetectionThread"),
        threading.Thread(target=run_surveillance_zones, args=(surv_queue, loop), name="SurveillanceZonesThread"),
        threading.Thread(target=run_zone_detection, args=(zonedet_queue, loop), name="ZoneDetectionThread"),
    ]

    for t in threads:
        t.start()

    # Keep event loop alive to let threads run indefinitely
    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main_async())
