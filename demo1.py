import asyncio
import threading
from oureyes.puller_bis import pull_stream
from oureyes.debug_timing import mark_stage
from models.fire_detection.fire_detection import fire_detection
from models.ppe_detection.ppe_detection import ppe_detection
from models.surveillance_zones.surveillance_zones import surveillance_zones
from models.time_count.time_count import time_count
from models.zone_analysis.zone_analysis import zone_analysis
from models.zone_detection.zone_detection import zone_detection

# Configuration
FPS = 25  # frames per second
INPUT_CAMS = ["cam1sub", "cam2sub"]  # Input streams for all models

# Model enable/disable configuration
MODEL_CONFIG = {
    "fire_detection": {
        "enabled": True,
        "function": fire_detection
    },
    "ppe_detection": {
        "enabled": False,
        "function": ppe_detection
    },
    "surveillance_zones": {
        "enabled": False,
        "function": surveillance_zones
    },
    "time_count": {
        "enabled": False,
        "function": time_count
    },
    "zone_analysis": {
        "enabled": False,
        "function": zone_analysis
    },
    "zone_detection": {
        "enabled": False,
        "function": zone_detection
    }
}


def sync_frame_generator(queue, loop, model_name, input_cam):
    """
    Converts an asyncio.Queue to a synchronous blocking generator.
    Keeps running even on errors, waiting for frames indefinitely.

    Also emits latency debug information (if enabled) for when a
    frame enters the model processing stage.
    """
    import time
    consecutive_errors = 0
    max_errors = 10
    last_warning_time = 0
    
    while True:
        try:
            # Get frame from queue (this will block until a frame is available)
            future = asyncio.run_coroutine_threadsafe(queue.get(), loop)
            # Use a reasonable timeout to prevent hanging
            frame = future.result(timeout=30.0)
            consecutive_errors = 0  # Reset error counter on success

            # Mark that this frame has entered the model stage
            mark_stage(
                stage="model_input",
                stream_name=f"{model_name}[{input_cam}]",
                frame=frame,
                pop=False,
            )

            yield frame
        except Exception as e:
            consecutive_errors += 1
            current_time = time.time()
            
            # Only log errors occasionally to avoid spam
            if consecutive_errors <= 3 or (current_time - last_warning_time) > 30:
                error_msg = str(e)
                if "timeout" not in error_msg.lower() or consecutive_errors <= 3:
                    print(f"⚠️ Error getting frame (attempt {consecutive_errors}): {error_msg}")
                last_warning_time = current_time
            
            # If too many consecutive errors, wait longer before retry
            if consecutive_errors > max_errors:
                if consecutive_errors == max_errors + 1:
                    print(f"⚠️ Many errors detected, waiting longer between retries...")
                time.sleep(2)
            else:
                time.sleep(0.5)
            
            # Continue loop to keep trying
            continue


def run_model(queue, loop, model_func, dest_cam, model_name, input_cam):
    """
    Run a model in a separate thread with automatic restart on failure.
    
    Args:
        queue: asyncio.Queue containing frames
        loop: asyncio event loop
        model_func: Model function to run
        dest_cam: Output camera name
        model_name: Name of the model for logging
        input_cam: Input camera name for logging
    """
    import time
    restart_delay = 5  # seconds to wait before restart
    
    while True:
        try:
            print(f"🚀 Starting {model_name} [{input_cam}] -> {dest_cam}")
            model_func(
                sync_frame_generator(queue, loop, model_name, input_cam),
                dest_cam=dest_cam,
                fps=FPS
            )
            # If model_func returns normally (shouldn't happen for infinite streams)
            print(f"⚠️ {model_name} [{input_cam}] finished normally, restarting in {restart_delay}s...")
        except KeyboardInterrupt:
            # Propagate keyboard interrupt
            print(f"🛑 {model_name} [{input_cam}] interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error in {model_name} [{input_cam}]: {e}")
            import traceback
            traceback.print_exc()
            print(f"🔄 Restarting {model_name} [{input_cam}] in {restart_delay}s...")
        
        # Wait before restarting (unless interrupted)
        try:
            time.sleep(restart_delay)
        except KeyboardInterrupt:
            break


async def main_async():
    """
    Main async function that pulls streams from both cameras and runs all enabled models.
    """
    loop = asyncio.get_running_loop()

    # For each enabled model and camera, subscribe separately so every
    # model receives the full frame stream. This avoids multiple
    # consumers competing for a single queue.
    model_queues = {}
    print(f"📡 Pulling streams from cameras: {', '.join(INPUT_CAMS)}\n")

    for input_cam in INPUT_CAMS:
        for model_name, config in MODEL_CONFIG.items():
            if config["enabled"]:
                print(f"📡 Connecting to {input_cam} for model {model_name}...")
                q = await pull_stream(input_cam)
                model_queues[(input_cam, model_name)] = q
                print(f"✅ Connected to {input_cam} for model {model_name}")

    # Create threads for each enabled model and each camera
    threads = []
    enabled_models = [name for name, config in MODEL_CONFIG.items() if config["enabled"]]
    total_threads = len(enabled_models) * len(INPUT_CAMS)
    
    print(f"\n🎯 Running {len(enabled_models)} models on {len(INPUT_CAMS)} cameras:")
    print(f"   Total processing threads: {total_threads}\n")
    
    for input_cam in INPUT_CAMS:
        for model_name, config in MODEL_CONFIG.items():
            if config["enabled"]:
                # Generate output camera name: model_name_camXsub
                dest_cam = f"{model_name}_{input_cam}"

                # Each model has its own queue subscription so it gets
                # all frames for this camera.
                q = model_queues[(input_cam, model_name)]

                thread = threading.Thread(
                    target=run_model,
                    args=(
                        q,
                        loop,
                        config["function"],
                        dest_cam,
                        model_name,
                        input_cam
                    ),
                    name=f"{model_name}_{input_cam}_Thread",
                    daemon=True
                )
                threads.append(thread)
                print(f"  ✅ {model_name:20s} [{input_cam:8s}] -> {dest_cam}")

    print(f"\n🚀 Starting {len(threads)} processing threads...\n")

    # Start all threads
    for thread in threads:
        thread.start()

    # Keep event loop alive and monitor threads
    print("⏳ Processing streams... (Press Ctrl+C to stop)\n")
    
    # Track which threads we've already warned about
    stopped_threads = set()
    
    try:
        while True:
            await asyncio.sleep(5)  # Check every 5 seconds instead of 1
            
            # Check thread status without spamming
            for thread in threads:
                if not thread.is_alive() and thread.name not in stopped_threads:
                    stopped_threads.add(thread.name)
                    print(f"⚠️ Thread {thread.name} has stopped (will auto-restart if configured)")
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        print("✅ All streams stopped")


if __name__ == "__main__":
    enabled_count = sum(1 for config in MODEL_CONFIG.values() if config["enabled"])
    print("=" * 70)
    print("🎬 OranextEYE - Multi-Model Processing Demo")
    print("=" * 70)
    print(f"Input Cameras:  {', '.join(INPUT_CAMS)}")
    print(f"Enabled Models: {enabled_count}")
    print(f"FPS:            {FPS}")
    print(f"Total Streams:  {enabled_count * len(INPUT_CAMS)}")
    print("=" * 70)
    
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n✅ Shutdown complete")
