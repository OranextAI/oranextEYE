# oureyes/__init__.py

from .puller import pull_stream
from .pusher import push_stream          # kept for backward compatibility
from .emitter import emit_detections
from .model_registry import get_yolo, get_yolo_lock, get_siglip, get_siglip_lock, DEVICE

__all__ = [
    "pull_stream",
    "push_stream",
    "emit_detections",
    "get_yolo",
    "get_yolo_lock",
    "get_siglip",
    "get_siglip_lock",
    "DEVICE",
]
