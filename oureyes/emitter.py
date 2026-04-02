"""
emitter.py — Lightweight detection emitter.

Singleton Socket.IO client shared across all model threads.
Sends JSON detection events to Node.js /camera namespace.
"""

import os
import time
import threading
from typing import Optional
import socketio
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:3000")

_sio: Optional[socketio.Client] = None
_init_lock = threading.Lock()   # only used during first-time init
_connected = False


def _init_client() -> socketio.Client:
    """Create and connect the singleton client. Called once."""
    global _sio, _connected

    sio = socketio.Client(
        reconnection=True,
        reconnection_attempts=0,
        reconnection_delay=2,
        logger=False,
        engineio_logger=False,
    )

    @sio.event(namespace='/camera')
    def connect():
        global _connected
        _connected = True
        print(f"[emitter] Connected to {BACKEND_URL}")

    @sio.event(namespace='/camera')
    def disconnect():
        global _connected
        _connected = False
        print("[emitter] Disconnected")

    @sio.event(namespace='/camera')
    def connect_error(data):
        global _connected
        _connected = False
        print(f"[emitter] Connection error: {data}")

    try:
        # polling first so the namespace handshake works, then upgrades to ws
        sio.connect(BACKEND_URL, namespaces=['/camera'],
                    transports=['polling', 'websocket'], wait_timeout=10)
    except Exception as e:
        print(f"[emitter] Could not connect: {e}")

    return sio


def _get_client() -> Optional[socketio.Client]:
    """Return the singleton, initialising it on first call."""
    global _sio
    if _sio is None:
        with _init_lock:
            if _sio is None:
                _sio = _init_client()
    return _sio


def emit_detections(stream_id: str, detections: list,
                    width: int = 0, height: int = 0) -> None:
    """
    Emit a detection event. Non-blocking — drops silently if not connected.
    Called from AI model threads at inference rate.
    """
    if not _connected:
        # Try to reconnect lazily without blocking the model thread
        client = _get_client()
        if not client or not client.connected:
            return

    payload = {
        "streamId": stream_id,
        "ts": int(time.time() * 1000),
        "width": width,
        "height": height,
        "detections": detections,
    }
    try:
        _sio.emit("detection", payload, namespace="/camera")
    except Exception:
        pass  # never block the model thread


def close():
    global _sio, _connected
    if _sio and _sio.connected:
        _sio.disconnect()
    _sio = None
    _connected = False
