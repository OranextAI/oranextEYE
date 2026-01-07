import subprocess
import sys
import time
import threading
import queue
import traceback
from oureyes.utils import build_rtsp_url

# Configuration constants
RECONNECT_DELAY = 2  # seconds between reconnection attempts
PROCESS_CHECK_INTERVAL = 1  # seconds between process health checks
FRAME_BUFFER_SIZE = 30  # Buffer size for frames
STDOUT_BUFFER_SIZE = 1000  # lines of stdout to keep for debugging

def push_stream(frames, width, height, fps, cam_name):
    """
    Push frames to RTSP using NVENC and TCP transport.
    Enhanced with automatic restart, process health monitoring, and robust error handling.
    Automatically restarts ffmpeg process if it crashes or connection is lost.
    
    Args:
        frames: Generator or iterator yielding frames (numpy arrays)
        width: Frame width in pixels
        height: Frame height in pixels
        fps: Frames per second
        cam_name: Destination camera name for RTSP URL
    """
    dest_url = build_rtsp_url(cam_name)
    print(f"🚀 Starting push stream to: {dest_url} ({width}x{height}@{fps}fps)")

    frame_queue = queue.Queue(maxsize=FRAME_BUFFER_SIZE)
    process = None
    process_lock = threading.Lock()
    stop_flag = threading.Event()
    last_error_log = []
    
    def build_ffmpeg_cmd():
        """Build ffmpeg command with appropriate settings."""
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-vf", "format=bgr0",
            "-c:v", "libx264",  # Use software encoding (more reliable)
            "-preset", "medium",
            "-tune", "zerolatency",
            "-b:v", "5M",
            "-g", "30",
            "-reconnect", "1",
            "-reconnect_at_eof", "1",
            "-reconnect_streamed", "1",
            "-reconnect_delay_max", "2",
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            "-muxdelay", "0",
            dest_url
        ]
        return cmd
    
    def read_stderr(pipe):
        """Read stderr from ffmpeg process for error logging."""
        try:
            for line in iter(pipe.readline, b''):
                if line:
                    error_msg = line.decode('utf-8', errors='ignore').strip()
                    if error_msg:
                        last_error_log.append(error_msg)
                        if len(last_error_log) > 10:
                            last_error_log.pop(0)
                        # Only print significant errors
                        error_lower = error_msg.lower()
                        if any(keyword in error_lower for keyword in ['error', 'failed', 'connection refused', 'network']):
                            print(f"⚠️ FFmpeg error for {cam_name}: {error_msg}")
        except Exception:
            pass
        finally:
            pipe.close()
    
    def start_ffmpeg_process():
        """Start a new ffmpeg process and return it."""
        with process_lock:
            # Close old process if exists
            if process is not None:
                try:
                    process.terminate()
                    process.wait(timeout=2)
                except:
                    try:
                        process.kill()
                    except:
                        pass
            
            cmd = build_ffmpeg_cmd()
            print(f"🔄 Starting FFmpeg process for {cam_name}...")
            
            try:
                new_process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0
                )
                
                # Start stderr reader thread
                stderr_thread = threading.Thread(
                    target=read_stderr,
                    args=(new_process.stderr,),
                    daemon=True
                )
                stderr_thread.start()
                
                return new_process
            except Exception as e:
                print(f"❌ Failed to start FFmpeg for {cam_name}: {e}")
                return None
    
    def frame_producer():
        """Read frames from generator and put them in queue."""
        try:
            for frame in frames:
                if stop_flag.is_set():
                    break
                
                # Ensure frame is correct size
                if frame.shape[:2] != (height, width):
                    print(f"⚠️ Frame size mismatch for {cam_name}: expected {(height, width)}, got {frame.shape[:2]}")
                    frame = frame[:height, :width] if frame.shape[0] >= height and frame.shape[1] >= width else frame
                
                # Put frame in queue (always copy to avoid modification issues)
                try:
                    frame_queue.put(frame.copy(), block=False)
                except queue.Full:
                    # Remove oldest frame and add new one
                    try:
                        frame_queue.get_nowait()
                        frame_queue.put(frame.copy(), block=False)
                    except:
                        pass
                        
        except Exception as e:
            print(f"⚠️ Error in frame producer for {cam_name}: {e}")
            if not stop_flag.is_set():
                print(traceback.format_exc())
        finally:
            print(f"✅ Frame producer finished for {cam_name}")
    
    def process_monitor():
        """Monitor ffmpeg process health and restart if needed."""
        nonlocal process
        
        while not stop_flag.is_set():
            time.sleep(PROCESS_CHECK_INTERVAL)
            
            with process_lock:
                if process is None or process.poll() is not None:
                    # Process died, restart it
                    if process is not None:
                        exit_code = process.poll()
                        print(f"⚠️ FFmpeg process died for {cam_name} (exit code: {exit_code})")
                        if last_error_log:
                            print(f"   Last errors: {'; '.join(last_error_log[-3:])}")
                    
                    if not stop_flag.is_set():
                        process = start_ffmpeg_process()
                        if process is None:
                            print(f"⚠️ Failed to restart FFmpeg for {cam_name}, retrying in {RECONNECT_DELAY}s...")
                            time.sleep(RECONNECT_DELAY)
    
    def frame_writer():
        """Write frames to ffmpeg process, restarting on errors."""
        nonlocal process
        
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while not stop_flag.is_set():
            try:
                # Get process reference
                with process_lock:
                    current_process = process
                
                if current_process is None:
                    time.sleep(RECONNECT_DELAY)
                    continue
                
                # Get frame from queue with timeout
                try:
                    frame = frame_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Write frame to process
                try:
                    if current_process.stdin and current_process.poll() is None:
                        current_process.stdin.write(frame.tobytes())
                        current_process.stdin.flush()
                        consecutive_errors = 0  # Reset error counter on success
                    else:
                        raise Exception("Process stdin not available or process dead")
                        
                except BrokenPipeError:
                    print(f"⚠️ Broken pipe for {cam_name}, process may have died")
                    consecutive_errors += 1
                    with process_lock:
                        if process == current_process:
                            process = None
                    time.sleep(0.1)
                    
                except Exception as e:
                    consecutive_errors += 1
                    print(f"⚠️ Error writing frame for {cam_name}: {e}")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"⚠️ Too many consecutive errors for {cam_name}, restarting process...")
                        consecutive_errors = 0
                        with process_lock:
                            if process == current_process:
                                process = None
                        time.sleep(RECONNECT_DELAY)
                    else:
                        time.sleep(0.1)
                        
            except KeyboardInterrupt:
                print(f"🛑 Streaming interrupted by user for {cam_name}")
                break
            except Exception as e:
                print(f"❌ Unexpected error in frame writer for {cam_name}: {e}")
                print(traceback.format_exc())
                time.sleep(RECONNECT_DELAY)
    
    # Start FFmpeg process
    process = start_ffmpeg_process()
    if process is None:
        print(f"❌ Failed to start initial FFmpeg process for {cam_name}")
        return
    
    print(f"✅ FFmpeg process started for {cam_name}")
    
    # Start threads
    producer_thread = threading.Thread(target=frame_producer, daemon=True, name=f"Producer-{cam_name}")
    writer_thread = threading.Thread(target=frame_writer, daemon=True, name=f"Writer-{cam_name}")
    monitor_thread = threading.Thread(target=process_monitor, daemon=True, name=f"Monitor-{cam_name}")
    
    producer_thread.start()
    writer_thread.start()
    monitor_thread.start()
    
    try:
        # Wait for producer to finish (frames generator exhausted)
        producer_thread.join()
        
        # Wait a bit for remaining frames to be written
        print(f"⏳ Waiting for remaining frames to be written for {cam_name}...")
        time.sleep(2)
        
    except KeyboardInterrupt:
        print(f"🛑 Streaming interrupted by user for {cam_name}")
    except Exception as e:
        print(f"⚠️ Error in push_stream for {cam_name}: {e}")
        print(traceback.format_exc())
    finally:
        # Signal stop
        stop_flag.set()
        
        # Wait for threads to finish
        writer_thread.join(timeout=2)
        monitor_thread.join(timeout=1)
        
        # Close process
        with process_lock:
            if process:
                try:
                    if process.stdin:
                        process.stdin.close()
                except:
                    pass
                try:
                    process.terminate()
                    process.wait(timeout=2)
                except:
                    try:
                        process.kill()
                        process.wait(timeout=1)
                    except:
                        pass
        
        print(f"✅ Stream stopped for {cam_name}")
