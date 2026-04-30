"""
stream_manager.py — RTSP-based frame broadcaster.

One RTSP connection per camera (via OpenCV/FFmpeg), shared across all AI models.
Uses a background thread for capture and thread-safe queue distribution.
WebRTC/aiortc is NOT used here — RTSP is faster and more reliable on the same server.
"""

import asyncio
import numpy as np
import os
import cv2
import time
import threading
from oureyes.utils import build_rtsp_url

# ── Silence OpenCV / FFmpeg noise ─────────────────────────────────────────
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["FFREPORT"] = "quiet"
os.environ["FFMPEG_LOGLEVEL"] = "quiet"
try:
    cv2.setLogLevel(0)
except AttributeError:
    pass

# ── Constants ──────────────────────────────────────────────────────────────
MAX_QUEUE_SIZE  = 1   # keep only the latest frame per subscriber
RECONNECT_DELAY = 2   # seconds between reconnect attempts
FRAME_TIMEOUT   = 10  # seconds without a frame before reconnecting


class StreamBroadcaster:
    """
    Maintains one RTSP connection per camera and fans out frames
    to all subscribed model queues.
    """

    def __init__(self, cam_name: str):
        self.cam_name = cam_name
        self.url = build_rtsp_url(cam_name)
        self._queues: list[asyncio.Queue] = []
        self._queues_lock = threading.Lock()
        self.connected_event = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop = None
        self._stop = False

    # ── Internal capture thread ────────────────────────────────────────────

    def _push_frame(self, frame: np.ndarray):
        """Thread-safe: push frame to all subscriber queues via the event loop."""
        with self._queues_lock:
            queues = list(self._queues)

        for queue in queues:
            img = np.copy(frame) if len(queues) > 1 else frame
            # Schedule put on the event loop thread — asyncio.Queue is not thread-safe
            asyncio.run_coroutine_threadsafe(self._put_nowait(queue, img), self._loop)

    @staticmethod
    async def _put_nowait(queue: asyncio.Queue, frame: np.ndarray):
        """Drop oldest frame and insert newest if queue is full."""
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            queue.put_nowait(frame)
        except asyncio.QueueFull:
            pass  # race condition safety

    def _capture_thread(self):
        """Blocking loop: open RTSP, read frames, push to queues, reconnect on failure."""
        while not self._stop:
            print(f"[stream_manager] Connecting to {self.url}")
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

            if not cap.isOpened():
                print(f"[stream_manager] ❌ Cannot open {self.url} — retrying in {RECONNECT_DELAY}s")
                time.sleep(RECONNECT_DELAY)
                continue

            print(f"[stream_manager] ✅ RTSP connected: {self.cam_name}")
            self._loop.call_soon_threadsafe(self.connected_event.set)

            last_frame_time = time.time()

            while not self._stop:
                ret, frame = cap.read()

                if not ret:
                    if time.time() - last_frame_time > FRAME_TIMEOUT:
                        print(f"[stream_manager] ⚠️ No frames for {FRAME_TIMEOUT}s from {self.cam_name} — reconnecting")
                        break
                    time.sleep(0.01)
                    continue

                last_frame_time = time.time()
                self._push_frame(frame)

            cap.release()

            if not self._stop:
                print(f"[stream_manager] 🔄 Reconnecting {self.cam_name} in {RECONNECT_DELAY}s...")
                time.sleep(RECONNECT_DELAY)

    # ── Public API ─────────────────────────────────────────────────────────

    async def start(self):
        """Spawn the capture thread (non-blocking)."""
        self._loop = asyncio.get_running_loop()
        t = threading.Thread(
            target=self._capture_thread,
            name=f"rtsp-{self.cam_name}",
            daemon=True,
        )
        t.start()

    def subscribe(self, max_queue_size: int = MAX_QUEUE_SIZE) -> asyncio.Queue:
        """Register a new subscriber and return its frame queue."""
        queue = asyncio.Queue(maxsize=max_queue_size)
        with self._queues_lock:
            self._queues.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue):
        """Remove a subscriber queue."""
        with self._queues_lock:
            if queue in self._queues:
                self._queues.remove(queue)

    def stop(self):
        self._stop = True


# ── Global broadcaster registry ───────────────────────────────────────────
_broadcasters: dict[str, StreamBroadcaster] = {}


async def get_stream(cam_name: str, max_wait_time: int = 60) -> asyncio.Queue:
    """
    Return a frame queue for the given camera.
    Creates and starts a StreamBroadcaster on first call; reuses it afterwards.
    """
    if cam_name not in _broadcasters:
        broadcaster = StreamBroadcaster(cam_name)
        _broadcasters[cam_name] = broadcaster
        await broadcaster.start()

    broadcaster = _broadcasters[cam_name]

    if not broadcaster.connected_event.is_set():
        try:
            await asyncio.wait_for(broadcaster.connected_event.wait(), timeout=max_wait_time)
        except asyncio.TimeoutError:
            raise RuntimeError(f"[stream_manager] Timeout connecting to {cam_name} after {max_wait_time}s")

    return broadcaster.subscribe()
