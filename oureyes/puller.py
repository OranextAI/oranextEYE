import asyncio
import aiohttp
import cv2
import numpy as np
import time
from aiortc import RTCPeerConnection, RTCSessionDescription
from oureyes.utils import build_webrtc_url

def pull_stream(cam_name):
    """
    Pull frames from WHEP stream built using cam_name.
    Example: cam_name="fakecamera" -> http://<HOST>:<PORT>/<cam_name>/whep
    """
    url = build_webrtc_url(cam_name)

    async def webrtc_worker(queue):
        async with aiohttp.ClientSession() as session:
            pc = RTCPeerConnection()
            pc.addTransceiver('video', direction='recvonly')

            @pc.on('track')
            def on_track(track):
                print(f"✅ Track received: {track.kind}")

                async def recv_loop():
                    while True:
                        frame = await track.recv()
                        img = frame.to_ndarray(format='bgr24')
                        await queue.put(img)
                asyncio.ensure_future(recv_loop())

            # Offer & answer handshake
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)

            async with session.post(url, headers={'Content-Type': 'application/sdp'}, data=pc.localDescription.sdp) as resp:
                answer_sdp = await resp.text()

            await pc.setRemoteDescription(RTCSessionDescription(sdp=answer_sdp, type='answer'))
            print(f"✅ WHEP session established for {cam_name}")

            # Keep connection alive
            while True:
                await asyncio.sleep(1)

    # Async runner
    async def frame_generator():
        queue = asyncio.Queue()
        while True:
            try:
                # Start WHEP worker
                task = asyncio.create_task(webrtc_worker(queue))
                while True:
                    frame = await queue.get()
                    yield frame
            except Exception as e:
                print(f"⚠️ Connection lost: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)

    # Run the async generator in sync context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    agen = frame_generator()
    while True:
        try:
            frame = loop.run_until_complete(agen.__anext__())
            yield frame
        except Exception as e:
            print(f"⚠️ Error pulling frame: {e}. Reconnecting...")
            time.sleep(5)
