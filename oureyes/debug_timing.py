import os
import time
import threading

"""Simple latency debugger for the OranextEYE pipeline.

Usage:
- Enable by setting environment variable EYE_LATENCY_DEBUG=1
- We attach a capture timestamp to each frame (by Python id),
  then measure latency at various stages (model input, pusher input, etc.).

This is lightweight and safe for production since it only adds
some dictionary lookups and optional print statements.
"""

ENABLE_DEBUG = os.getenv("EYE_LATENCY_DEBUG", "0") == "1"

_lock = threading.Lock()
# Map: id(frame) -> (t_capture, source_name)
_frame_times = {}


def register_capture(source_name: str, frame) -> None:
    """Register the capture time of a frame.

    source_name: e.g. "cam1sub" (camera path).
    frame: numpy array representing the frame.
    """
    if not ENABLE_DEBUG:
        return
    t = time.time()
    fid = id(frame)
    with _lock:
        _frame_times[fid] = (t, source_name)


def mark_stage(stage: str, stream_name: str, frame, pop: bool = False) -> None:
    """Mark a processing stage for a frame and print latency from capture.

    stage: logical stage name, e.g. "model_input", "pusher_input".
    stream_name: something meaningful like "fire_detection[cam1sub]".
    frame: numpy array for which we want to compute latency.
    pop: if True, remove the frame entry from the internal map (final stage).
    """
    if not ENABLE_DEBUG:
        return

    now = time.time()
    fid = id(frame)
    with _lock:
        entry = _frame_times.get(fid)
        if pop and entry is not None:
            _frame_times.pop(fid, None)

    if entry is None:
        # We didn't see this frame at capture; still print something.
        print(f"[LATENCY] {stage} {stream_name}: capture_ts_unknown")
    else:
        t_capture, src = entry
        age_ms = (now - t_capture) * 1000.0
        print(f"[LATENCY] {stage} {stream_name}: from_capture={age_ms:.1f}ms src={src}")
