"""
model_registry.py — Global model cache with per-model inference locks.

Loads each AI model exactly ONCE. Multiple cameras sharing the same model
use a per-model lock to serialize inference calls — Ultralytics YOLO's
internal predictor state is not thread-safe for concurrent calls on the
same object.

This means two cameras running the same model will take turns on inference
(each call is ~30-100ms on GPU) — still far cheaper than loading the model
twice (which would double VRAM usage).
"""

import threading
from typing import Any, Dict
import torch

_lock  = threading.Lock()          # protects _cache writes
_cache: Dict[str, Any] = {}
_model_locks: Dict[str, threading.Lock] = {}  # per-model inference lock

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[model_registry] Using device: {DEVICE}")


def _get_inference_lock(key: str) -> threading.Lock:
    """Return (creating if needed) the per-model inference lock."""
    if key not in _model_locks:
        with _lock:
            if key not in _model_locks:
                _model_locks[key] = threading.Lock()
    return _model_locks[key]


def get_yolo(model_path: str):
    """
    Return a cached YOLO model.
    Callers must acquire get_yolo_lock(model_path) around each inference call.
    """
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


def get_yolo_lock(model_path: str) -> threading.Lock:
    """Return the inference lock for a YOLO model path."""
    return _get_inference_lock(f"yolo:{model_path}")


def get_siglip(model_name: str):
    """Return a cached SigLIP (model, processor, enabled) tuple."""
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
                    dummy  = Image.new("RGB", (224, 224), 0)
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


def get_siglip_lock(model_name: str) -> threading.Lock:
    """Return the inference lock for a SigLIP model."""
    return _get_inference_lock(f"siglip:{model_name}")
