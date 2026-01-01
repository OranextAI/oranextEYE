import asyncio
import aiohttp
import cv2
import numpy as np
import time
import traceback
from aiortc import RTCPeerConnection, RTCSessionDescription
from oureyes.utils import build_webrtc_url

# Configuration constants
RECONNECT_DELAY = 2  # seconds between reconnection attempts
MAX_QUEUE_SIZE = 30  # prevent memory buildup
FRAME_TIMEOUT = 10  # seconds without frames before considering connection lost
CONNECTION_TIMEOUT = 30  # seconds for initial connection

def pull_stream(cam_name):
    """
    Pull frames from WHEP stream built using cam_name.
    Enhanced with robust reconnection, connection monitoring, and queue management.
    Example: cam_name="fakecamera" -> http://<HOST>:<PORT>/<cam_name>/whep
    
    Args:
        cam_name: Name of the camera stream to pull
        
    Yields:
        numpy.ndarray: BGR frames from the stream
    """
    url = build_webrtc_url(cam_name)
    print(f"🎬 Starting pull stream for: {cam_name} ({url})")

    async def webrtc_worker(queue, pc_event, connected_event):
        """
        WebRTC worker that handles connection, frame reception, and reconnection.
        """
        pc = None
        recv_task = None
        last_frame_time = None
        
        while True:
            try:
                # Clear previous connection state
                if pc:
                    try:
                        await pc.close()
                    except:
                        pass
                
                pc_event.clear()
                connected_event.clear()
                
                # Create new peer connection
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=CONNECTION_TIMEOUT)) as session:
                    pc = RTCPeerConnection()
                    pc_event.set()
                    
                    connection_state = {'failed': False}
                    
                    @pc.on('track')
                    def on_track(track):
                        nonlocal recv_task, last_frame_time
                        print(f"✅ Track received from {cam_name}: {track.kind}")
                        
                        async def recv_loop():
                            nonlocal last_frame_time
                            while True:
                                try:
                                    # Wait for frame with timeout
                                    frame = await asyncio.wait_for(track.recv(), timeout=FRAME_TIMEOUT)
                                    img = frame.to_ndarray(format='bgr24')
                                    last_frame_time = time.time()
                                    
                                    # Put frame in queue, drop old frames if queue is full
                                    try:
                                        queue.put_nowait(img)
                                    except asyncio.QueueFull:
                                        # Remove oldest frame and add new one
                                        try:
                                            queue.get_nowait()
                                            queue.put_nowait(img)
                                        except:
                                            pass
                                except asyncio.TimeoutError:
                                    # Check if connection is still alive
                                    if pc.iceConnectionState in ('failed', 'disconnected', 'closed'):
                                        print(f"⚠️ Connection state: {pc.iceConnectionState} for {cam_name}")
                                        break
                                    # Check if no frames received for too long
                                    if last_frame_time and (time.time() - last_frame_time) > FRAME_TIMEOUT:
                                        print(f"⚠️ No frames received for {FRAME_TIMEOUT}s from {cam_name}")
                                        break
                                except Exception as e:
                                    print(f"⚠️ Error receiving frame from {cam_name}: {e}")
                                    break
                        
                        recv_task = asyncio.create_task(recv_loop())
                    
                    # Monitor ICE connection state
                    @pc.on('iceconnectionstatechange')
                    def on_iceconnectionstatechange():
                        state = pc.iceConnectionState
                        print(f"📡 ICE connection state for {cam_name}: {state}")
                        if state in ('failed', 'disconnected', 'closed'):
                            connection_state['failed'] = True
                    
                    @pc.on('connectionstatechange')
                    def on_connectionstatechange():
                        state = pc.connectionState
                        print(f"🔌 Connection state for {cam_name}: {state}")
                        if state == 'failed':
                            connection_state['failed'] = True
                    
                    # Setup transceiver
                    pc.addTransceiver('video', direction='recvonly')
                    
                    # Create and send offer (WHEP)
                    offer = await pc.createOffer()
                    await pc.setLocalDescription(offer)
                    
                    # Send offer to server
                    try:
                        async with session.post(
                            url,
                            headers={'Content-Type': 'application/sdp'},
                            data=pc.localDescription.sdp,
                            timeout=aiohttp.ClientTimeout(total=CONNECTION_TIMEOUT)
                        ) as resp:
                            # Accept both 200 (OK) and 201 (Created) as success
                            if resp.status not in (200, 201):
                                raise Exception(f"WHEP server returned status {resp.status}")
                            answer_sdp = await resp.text()
                    except asyncio.TimeoutError:
                        raise Exception(f"Connection timeout to {url}")
                    
                    # Set remote description
                    await pc.setRemoteDescription(
                        RTCSessionDescription(sdp=answer_sdp, type='answer')
                    )
                    
                    print(f"✅ WHEP session established for {cam_name}")
                    connected_event.set()
                    last_frame_time = time.time()
                    
                    # Monitor connection health
                    while True:
                        await asyncio.sleep(1)
                        
                        # Check connection state
                        if connection_state['failed']:
                            print(f"⚠️ Connection failed detected for {cam_name}")
                            break
                        
                        # Check if recv_task is still running
                        if recv_task and recv_task.done():
                            print(f"⚠️ Receive task stopped for {cam_name}")
                            try:
                                await recv_task  # Get exception if any
                            except Exception as e:
                                print(f"⚠️ Receive task error: {e}")
                            break
                        
                        # Check for no frames timeout
                        if last_frame_time and (time.time() - last_frame_time) > FRAME_TIMEOUT * 2:
                            print(f"⚠️ No frames for {FRAME_TIMEOUT * 2}s, reconnecting {cam_name}")
                            break
                        
                        # Check ICE state
                        if pc.iceConnectionState in ('failed', 'disconnected', 'closed'):
                            print(f"⚠️ ICE state indicates disconnect for {cam_name}")
                            break
                
                # Cleanup
                if recv_task and not recv_task.done():
                    recv_task.cancel()
                    try:
                        await recv_task
                    except asyncio.CancelledError:
                        pass
                
                if pc:
                    try:
                        await pc.close()
                    except:
                        pass
                
            except Exception as e:
                print(f"❌ Error in webrtc_worker for {cam_name}: {e}")
                print(traceback.format_exc())
                if pc:
                    try:
                        await pc.close()
                    except:
                        pass
            
            # Wait before reconnecting
            print(f"🔄 Reconnecting {cam_name} in {RECONNECT_DELAY} seconds...")
            await asyncio.sleep(RECONNECT_DELAY)

    # Async frame generator
    async def frame_generator():
        queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        pc_event = asyncio.Event()
        connected_event = asyncio.Event()
        
        # Start worker task
        worker_task = None
        
        while True:
            try:
                # Start or restart worker
                if worker_task is None or worker_task.done():
                    worker_task = asyncio.create_task(
                        webrtc_worker(queue, pc_event, connected_event)
                    )
                
                # Wait for connection to be established
                try:
                    await asyncio.wait_for(connected_event.wait(), timeout=60)
                except asyncio.TimeoutError:
                    print(f"⚠️ Connection timeout for {cam_name}, retrying...")
                    continue
                
                # Yield frames from queue
                while True:
                    try:
                        # Get frame with timeout to check connection
                        frame = await asyncio.wait_for(queue.get(), timeout=5.0)
                        yield frame
                    except asyncio.TimeoutError:
                        # Check if connection is still alive
                        if not connected_event.is_set():
                            print(f"⚠️ Connection lost for {cam_name}, waiting for reconnect...")
                            break
                        # Continue waiting for frames
                        continue
                        
            except Exception as e:
                print(f"⚠️ Error in frame_generator for {cam_name}: {e}")
                print(traceback.format_exc())
                await asyncio.sleep(RECONNECT_DELAY)

    # Run the async generator in sync context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    agen = frame_generator()
    
    consecutive_errors = 0
    max_consecutive_errors = 10
    
    while True:
        try:
            frame = loop.run_until_complete(agen.__anext__())
            consecutive_errors = 0  # Reset error counter on success
            yield frame
        except StopAsyncIteration:
            print(f"⚠️ Stream ended for {cam_name}, restarting...")
            time.sleep(RECONNECT_DELAY)
        except Exception as e:
            consecutive_errors += 1
            print(f"⚠️ Error pulling frame from {cam_name}: {e}")
            if consecutive_errors >= max_consecutive_errors:
                print(f"❌ Too many consecutive errors ({consecutive_errors}) for {cam_name}, waiting longer...")
                time.sleep(RECONNECT_DELAY * 2)
                consecutive_errors = 0
            else:
                time.sleep(RECONNECT_DELAY)