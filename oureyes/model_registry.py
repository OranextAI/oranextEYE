"""
model_registry.py — Global model cache.

Loads each AI model exactly ONCE, no matter how many cameras or threads
use it. All threads share the same model instance.

Thread safety: model inference (YOLO / SigLIP) is stateless for a single
forward pass, so sharing the loaded model object across threads is safe as
long as each thread builds its own input tensors (which they do).

Usage:
    from oureyes.model_registry import get_yolo, get_siglip

    model = get_yolo("/path/to/weights.pt")   # loaded once, cached forever
"""

import threading
from typing import Any, Dict
import torch

_lock = threading.Lock()
_cache: Dict[str, Any] = {}

# Auto-select device once at import time
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[model_registry] Using device: {DEVICE}")


def get_yolo(model_path: str):
    """Return a cached YOLO model, loading it on first call."""
    key = f"yolo:{model_path}"
    if key not in _cache:
        with _lock:
            if key not in _cache:
                from ultralytics import YOLO
                print(f"[model_registry] Loading YOLO: {model_path} on {DEVICE}")
                model = YOLO(model_path)
                if DEVICE == "cuda":
                    model.to(DEVICE)
                _cache[key] = model
                print(f"[model_registry] YOLO loaded: {model_path}")
    return _cache[key]


def get_siglip(model_name: str):
    """Return a cached SigLIP (model, processor, enabled) tuple, loading on first call."""
    key = f"siglip:{model_name}"
    if key not in _cache:
        with _lock:
            if key not in _cache:
                from transformers import AutoImageProcessor, SiglipForImageClassification
                from PIL import Image
                print(f"[model_registry] Loading SigLIP: {model_name} on {DEVICE}")
                model = SiglipForImageClassification.from_pretrained(model_name)
                processor = AutoImageProcessor.from_pretrained(model_name)
                if DEVICE == "cuda":
                    model = model.to(DEVICE)
                model.eval()
                try:
                    dummy = Image.new("RGB", (224, 224), 0)
                    inputs = processor(images=dummy, return_tensors="pt")
                    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                    with torch.no_grad():
                        model(**inputs)
                    print(f"[model_registry] SigLIP ready: {model_name}")
                    _cache[key] = (model, processor, True)
                except Exception as e:
                    print(f"[model_registry] SigLIP disabled ({e}): {model_name}")
                    _cache[key] = (model, processor, False)
    return _cache[key]


def get_trackzone(model_path: str, region):
    """Return a cached TrackZone instance."""
    key = f"trackzone:{model_path}"
    if key not in _cache:
        with _lock:
            if key not in _cache:
                from ultralytics import solutions
                print(f"[model_registry] Loading TrackZone: {model_path}")
                _cache[key] = solutions.TrackZone(
                    show=False, region=region, model=model_path, verbose=False
                )
                print(f"[model_registry] TrackZone loaded: {model_path}")
    return _cache[key]
