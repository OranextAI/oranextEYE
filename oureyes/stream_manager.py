import asyncio
import aiohttp
import numpy as np
import os
import cv2
from aiortc import RTCPeerConnection, RTCSessionDescription
from oureyes.utils import build_webrtc_url

# ======================================================
# ✅ Silence OpenCV + FFmpeg logs across all versions
# ======================================================
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["FFREPORT"] = "quiet"
os.environ["FFMPEG_LOGLEVEL"] = "quiet"

try:
    cv2.setLogLevel(0)
except AttributeError:
    pass

# ======================================================
# Stream Broadcaster
# ======================================================
class StreamBroadcaster:
    def __init__(self, cam_name):
        self.cam_name = cam_name
        self.url = build_webrtc_url(cam_name)
        self.queues = []
        self.pc = None
        self.recv_task = None
        self.connected_event = asyncio.Event()
        self._stop = False

    async def start(self):
        """Establish and maintain a WebRTC stream connection."""
        while not self._stop:
            try:
                async with aiohttp.ClientSession() as session:
                    self.pc = RTCPeerConnection()
                    self.pc.addTransceiver('video', direction='recvonly')

                    # ----------------------------------------
                    # Handle incoming video track
                    # ----------------------------------------
                    @self.pc.on('track')
                    def on_track(track):
                        print(f"✅ Track received from camera {self.cam_name}: {track.kind}")

                        async def recv_loop():
                            while True:
                                try:
                                    frame = await track.recv()
                                    img = frame.to_ndarray(format='bgr24')
                                    for queue in self.queues:
                                        try:
                                            queue.put_nowait(np.copy(img))  # ✅ each consumer gets its own frame
                                        except asyncio.QueueFull:
                                            pass
                                except Exception:
                                    break  # exit loop if stream stops

                        self.recv_task = asyncio.create_task(recv_loop())

                    # ----------------------------------------
                    # ICE Connection State Changes
                    # ----------------------------------------
                    @self.pc.on('iceconnectionstatechange')
                    def on_iceconnectionstatechange():
                        state = self.pc.iceConnectionState
                        if state in ('failed', 'disconnected', 'closed'):
                            print(f"⚠️ Connection lost for {self.cam_name}, will reconnect...")
                            asyncio.create_task(self.pc.close())

                    # ----------------------------------------
                    # Create and send offer (WHEP)
                    # ----------------------------------------
                    offer = await self.pc.createOffer()
                    await self.pc.setLocalDescription(offer)

                    async with session.post(
                        self.url,
                        headers={'Content-Type': 'application/sdp'},
                        data=self.pc.localDescription.sdp
                    ) as resp:
                        answer_sdp = await resp.text()

                    await self.pc.setRemoteDescription(
                        RTCSessionDescription(sdp=answer_sdp, type='answer')
                    )

                    print(f"✅ WHEP session established for camera {self.cam_name}")
                    self.connected_event.set()

                    # ----------------------------------------
                    # Wait while connected
                    # ----------------------------------------
                    while self.pc.iceConnectionState not in ('failed', 'disconnected', 'closed'):
                        await asyncio.sleep(1)

                    await self.pc.close()
                    self.pc = None

            except Exception as e:
                print(f"❌ Connection error for camera {self.cam_name}: {e}")

            # ----------------------------------------
            # Reconnect handling
            # ----------------------------------------
            print(f"🔄 Reconnecting {self.cam_name} in 5 seconds...")
            self.connected_event.clear()
            await asyncio.sleep(5)

    # ======================================================
    # Public methods
    # ======================================================
    def subscribe(self, max_queue_size=10):
        """Subscribe to the stream (each subscriber gets its own queue)."""
        queue = asyncio.Queue(maxsize=max_queue_size)
        self.queues.append(queue)
        return queue

    async def stop(self):
        """Stop streaming gracefully."""
        self._stop = True
        if self.pc:
            await self.pc.close()


# ======================================================
# Global manager
# ======================================================
broadcasters = {}

async def get_stream(cam_name):
    """Return a frame queue for the requested camera stream."""
    if cam_name not in broadcasters:
        broadcaster = StreamBroadcaster(cam_name)
        broadcasters[cam_name] = broadcaster
        asyncio.create_task(broadcaster.start())
        await broadcaster.connected_event.wait()
    else:
        broadcaster = broadcasters[cam_name]

    return broadcaster.subscribe()
