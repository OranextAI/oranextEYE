import asyncio
import aiohttp
import numpy as np
import os
import cv2
import time
from aiortc import RTCPeerConnection, RTCSessionDescription
from oureyes.utils import build_webrtc_url
from oureyes.debug_timing import register_capture

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

# Configuration constants
MAX_QUEUE_SIZE = 30  # Buffer size for frames

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
        RECONNECT_DELAY = 2
        FRAME_TIMEOUT = 10
        
        while not self._stop:
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                    # Clean up old connection
                    if self.pc:
                        try:
                            await self.pc.close()
                        except:
                            pass
                        self.pc = None
                    
                    if self.recv_task and not self.recv_task.done():
                        self.recv_task.cancel()
                        try:
                            await self.recv_task
                        except asyncio.CancelledError:
                            pass
                    
                    self.pc = RTCPeerConnection()
                    self.connected_event.clear()
                    
                    connection_state = {'failed': False}
                    last_frame_time = None

                    # ----------------------------------------
                    # Handle incoming video track
                    # ----------------------------------------
                    @self.pc.on('track')
                    def on_track(track):
                        print(f"✅ Track received from camera {self.cam_name}: {track.kind}")

                        async def recv_loop():
                            nonlocal last_frame_time
                            # Diagnostics: count how many frames we drop because
                            # subscriber queues are full.
                            dropped_frames = 0
                            while True:
                                try:
                                    frame = await asyncio.wait_for(track.recv(), timeout=FRAME_TIMEOUT)
                                    img = frame.to_ndarray(format='bgr24')
                                    last_frame_time = time.time()
                                    
                                    # Distribute to all queues - always copy to avoid modification conflicts
                                    queues_to_remove = []
                                    
                                    for queue in self.queues:
                                        try:
                                            # Always copy frame for each queue to ensure independence
                                            frame_to_put = np.copy(img)
                                            # Register capture time for latency debugging
                                            register_capture(self.cam_name, frame_to_put)
                                            queue.put_nowait(frame_to_put)
                                        except asyncio.QueueFull:
                                            # Queue full, drop oldest frame and add new one
                                            dropped_frames += 1
                                            if dropped_frames in (1, 100, 1000) or dropped_frames % 2000 == 0:
                                                print(f"⚠️ [{self.cam_name}] Broadcaster dropped {dropped_frames} frame(s) due to slow subscribers")
                                            try:
                                                queue.get_nowait()
                                                frame_to_put = np.copy(img)
                                                # Register capture time again for the replacement frame
                                                register_capture(self.cam_name, frame_to_put)
                                                queue.put_nowait(frame_to_put)
                                            except:
                                                pass
                                        except Exception:
                                            # Queue may have been removed
                                            queues_to_remove.append(queue)
                                    for q in queues_to_remove:
                                        if q in self.queues:
                                            self.queues.remove(q)
                                            
                                except asyncio.TimeoutError:
                                    # Check if connection is still alive
                                    if self.pc and self.pc.iceConnectionState in ('failed', 'disconnected', 'closed'):
                                        break
                                    # Check if no frames for too long
                                    if last_frame_time and (time.time() - last_frame_time) > FRAME_TIMEOUT:
                                        print(f"⚠️ No frames for {FRAME_TIMEOUT}s from {self.cam_name}")
                                        break
                                    continue
                                except Exception as e:
                                    print(f"⚠️ Error receiving frame from {self.cam_name}: {e}")
                                    break

                        self.recv_task = asyncio.create_task(recv_loop())

                    # ----------------------------------------
                    # ICE Connection State Changes
                    # ----------------------------------------
                    @self.pc.on('iceconnectionstatechange')
                    def on_iceconnectionstatechange():
                        state = self.pc.iceConnectionState
                        print(f"📡 ICE connection state for {self.cam_name}: {state}")
                        if state in ('failed', 'disconnected', 'closed'):
                            connection_state['failed'] = True
                            self.connected_event.clear()

                    @self.pc.on('connectionstatechange')
                    def on_connectionstatechange():
                        state = self.pc.connectionState
                        print(f"🔌 Connection state for {self.cam_name}: {state}")
                        if state == 'failed':
                            connection_state['failed'] = True
                            self.connected_event.clear()

                    # ----------------------------------------
                    # Setup transceiver
                    # ----------------------------------------
                    self.pc.addTransceiver('video', direction='recvonly')

                    # ----------------------------------------
                    # Create and send offer (WHEP)
                    # ----------------------------------------
                    offer = await self.pc.createOffer()
                    await self.pc.setLocalDescription(offer)

                    try:
                        async with session.post(
                            self.url,
                            headers={'Content-Type': 'application/sdp'},
                            data=self.pc.localDescription.sdp,
                            timeout=aiohttp.ClientTimeout(total=30)
                        ) as resp:
                            # Accept both 200 (OK) and 201 (Created) as success
                            if resp.status not in (200, 201):
                                raise Exception(f"WHEP server returned status {resp.status}")
                            answer_sdp = await resp.text()
                    except asyncio.TimeoutError:
                        raise Exception(f"Connection timeout to {self.url}")

                    await self.pc.setRemoteDescription(
                        RTCSessionDescription(sdp=answer_sdp, type='answer')
                    )

                    print(f"✅ WHEP session established for camera {self.cam_name}")
                    self.connected_event.set()
                    last_frame_time = time.time()

                    # ----------------------------------------
                    # Monitor connection health
                    # ----------------------------------------
                    while not self._stop:
                        await asyncio.sleep(1)
                        
                        # Check connection state
                        if connection_state['failed']:
                            print(f"⚠️ Connection failed detected for {self.cam_name}")
                            break
                        
                        # Check if recv_task is still running
                        if self.recv_task and self.recv_task.done():
                            print(f"⚠️ Receive task stopped for {self.cam_name}")
                            try:
                                await self.recv_task  # Get exception if any
                            except Exception as e:
                                print(f"⚠️ Receive task error: {e}")
                            break
                        
                        # Check for no frames timeout
                        if last_frame_time and (time.time() - last_frame_time) > FRAME_TIMEOUT * 2:
                            print(f"⚠️ No frames for {FRAME_TIMEOUT * 2}s, reconnecting {self.cam_name}")
                            break
                        
                        # Check ICE state
                        if self.pc.iceConnectionState in ('failed', 'disconnected', 'closed'):
                            print(f"⚠️ ICE state indicates disconnect for {self.cam_name}")
                            break

                    # Cleanup
                    if self.recv_task and not self.recv_task.done():
                        self.recv_task.cancel()
                        try:
                            await self.recv_task
                        except asyncio.CancelledError:
                            pass

                    if self.pc:
                        try:
                            await self.pc.close()
                        except:
                            pass
                        self.pc = None

            except Exception as e:
                print(f"❌ Connection error for camera {self.cam_name}: {e}")
                import traceback
                print(traceback.format_exc())

            # ----------------------------------------
            # Reconnect handling
            # ----------------------------------------
            if not self._stop:
                print(f"🔄 Reconnecting {self.cam_name} in {RECONNECT_DELAY} seconds...")
                self.connected_event.clear()
                await asyncio.sleep(RECONNECT_DELAY)

    # ======================================================
    # Public methods
    # ======================================================
    def subscribe(self, max_queue_size=None):
        """Subscribe to the stream (each subscriber gets its own queue)."""
        if max_queue_size is None:
            max_queue_size = MAX_QUEUE_SIZE
        queue = asyncio.Queue(maxsize=max_queue_size)
        self.queues.append(queue)
        return queue
    
    def unsubscribe(self, queue):
        """Unsubscribe from the stream by removing the queue."""
        if queue in self.queues:
            self.queues.remove(queue)

    async def stop(self):
        """Stop streaming gracefully."""
        self._stop = True
        if self.pc:
            await self.pc.close()


# ======================================================
# Global manager
# ======================================================
broadcasters = {}

async def get_stream(cam_name, max_wait_time=60):
    """Return a frame queue for the requested camera stream.
    
    Args:
        cam_name: Name of the camera stream
        max_wait_time: Maximum seconds to wait for connection (default: 60)
    
    Returns:
        asyncio.Queue: Queue that will receive frames from the stream
    """
    if cam_name not in broadcasters:
        broadcaster = StreamBroadcaster(cam_name)
        broadcasters[cam_name] = broadcaster
        asyncio.create_task(broadcaster.start())
        
        # Wait for connection with timeout
        try:
            await asyncio.wait_for(broadcaster.connected_event.wait(), timeout=max_wait_time)
        except asyncio.TimeoutError:
            print(f"⚠️ Timeout waiting for connection to {cam_name}")
            raise Exception(f"Failed to connect to {cam_name} within {max_wait_time} seconds")
    else:
        broadcaster = broadcasters[cam_name]
        # Wait for connection if not already connected
        if not broadcaster.connected_event.is_set():
            try:
                await asyncio.wait_for(broadcaster.connected_event.wait(), timeout=max_wait_time)
            except asyncio.TimeoutError:
                print(f"⚠️ Timeout waiting for reconnection to {cam_name}")

    return broadcaster.subscribe()
