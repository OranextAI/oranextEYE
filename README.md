# OranextEYE — AI Vision Pipeline

Python library for pulling camera streams, running AI inference, and pushing detection events to the web app via Socket.IO — with zero video re-encoding.

---

## How It Works

```
MediaMTX (WebRTC) ──▶ stream_manager.py (one connection per camera)
                              │
                    fan-out to model queues
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        fire_detection   zone_analysis   ppe_detection  ...
              │               │               │
              └───────────────┴───────────────┘
                              │
                    emit_detections() via Socket.IO
                              │
                    Node.js /camera namespace
                              │
                    Angular canvas overlay
                    (raw HLS video untouched)
```

**Key optimisations:**
- Models loaded **once** via `model_registry.py`, shared across all threads
- One WebRTC connection per camera regardless of how many models run on it
- No FFmpeg, no encoding — detections are JSON events

---

## Structure

```
oranextEYE/
├── oureyes/
│   ├── emitter.py          # Singleton Socket.IO client → Node.js
│   ├── model_registry.py   # Load-once model cache (YOLO, SigLIP, TrackZone)
│   ├── stream_manager.py   # One WebRTC connection per camera, fan-out
│   ├── puller.py           # Legacy sync puller (kept for compat)
│   ├── puller_bis.py       # Async puller used by demo1.py
│   ├── pusher.py           # Legacy FFmpeg pusher (kept for compat)
│   ├── notifier.py         # Kafka alert producer
│   └── utils.py            # URL builders
├── models/
│   ├── fire_detection/     # SigLIP fire/smoke classifier
│   ├── zone_detection/     # YOLO object detection in zones
│   ├── zone_analysis/      # YOLO worker presence + time tracking
│   ├── ppe_detection/      # YOLO PPE compliance
│   ├── surveillance_zones/ # YOLO zone surveillance + absence timer
│   └── time_count/         # YOLO TrackZone working/not-working
├── media-config/
│   └── mediamtx.yml        # MediaMTX config (WebRTC, HLS, RTSP)
├── demo1.py                # Main runner
├── setup.py
└── .env
```

---

## Setup

### 1. Create conda environment

```bash
conda create -n nefzi_world python=3.9
conda activate nefzi_world
```

### 2. Install the library

```bash
cd oranextEYE
pip install -e .
pip install "python-socketio[client]"
```

### 3. Configure `.env`

```env
HOST=<mediamtx-server-ip>
PORT=8554
HOST_WEBRTC=<mediamtx-server-ip>
PORT_WEBRTC=8889
KAFKA_BOOTSTRAP_SERVERS=<kafka-ip>:19092
BACKEND_URL=http://127.0.0.1:3000
```

---

## Running

### Start MediaMTX

```bash
./mediamtx media-config/mediamtx.yml
```

### Run the AI pipeline

```bash
conda activate nefzi_world
python demo1.py
```

### Configure which models run

Edit `demo1.py`:

```python
INPUT_CAMS = ["cam2sub"]   # one entry per camera

MODEL_CONFIG = {
    "fire_detection":    {"enabled": True,  "fn": fire_detection},
    "zone_analysis":     {"enabled": True,  "fn": zone_analysis},
    "ppe_detection":     {"enabled": False, "fn": ppe_detection},
    "surveillance_zones":{"enabled": False, "fn": surveillance_zones},
    "time_count":        {"enabled": False, "fn": time_count},
    "zone_detection":    {"enabled": False, "fn": zone_detection},
}
```

Each enabled model gets its own thread and its own frame queue. The camera connection is shared.

---

## Fake Cameras — Video Preparation

WebRTC requires H.264 **Baseline profile** (no B-frames). Most downloaded videos use
Main/High profile with B-frames, which causes mediamtx to reject the stream with:

```
WebRTC doesn't support H264 streams with B-frames
```

The solution: re-encode once, save a `_ready.mp4`, then stream-copy forever with zero CPU.

### Step 1 — Check the video

```bash
ffprobe -v quiet -select_streams v:0 \
  -show_entries stream=codec_name,width,height,pix_fmt \
  -of default ~/videos/your-video.mp4
```

### Step 2 — Re-encode once (720p, Baseline, no B-frames)

```bash
ffmpeg -i ~/videos/your-video.mp4 \
  -vf scale=1280:720 \
  -c:v libx264 -preset veryfast \
  -profile:v baseline -level 3.1 \
  -x264-params "bframes=0:ref=1" \
  -g 30 -pix_fmt yuv420p -an \
  ~/videos/your-video_ready.mp4
```

If the video is already 720p, drop the `-vf scale=1280:720` line.

### Step 3 — Verify the output

```bash
ffprobe -v quiet -select_streams v:0 \
  -show_entries stream=codec_name,profile,width,height \
  -of default ~/videos/your-video_ready.mp4
# Expected: codec_name=h264, profile=Constrained Baseline, 1280x720
```

### Step 4 — Use in mediamtx.yml with `-c:v copy`

```yaml
  my_fakecam:
    source: publisher
    overridePublisher: yes
    runOnInit: >
      ffmpeg -re -stream_loop -1 -i /videos/your-video_ready.mp4
      -c:v copy -bsf:v h264_mp4toannexb -an
      -f rtsp rtsp://127.0.0.1:8554/my_fakecam
    runOnInitRestart: yes
    record: yes
    recordPath: /recordings/%path/%Y-%m-%d_%H-%M-%S-%f
```

`-c:v copy` — zero CPU re-encoding on every loop.  
`-bsf:v h264_mp4toannexb` — **required**: converts SPS/PPS from MP4 container format (`avcC`) to inline Annex B format. Without this, `aiortc` (used by the AI pipeline) connects but receives no decodable frames.

Then apply:
```bash
bash media-config/sync-config.sh
```

---

## Writing a New Model

```python
# oranextEYE/models/my_model/my_model.py

from oureyes.emitter import emit_detections
from oureyes.model_registry import get_yolo

def my_model(frames, dest_cam: str, fps: int):
    MODEL_PATH = "path/to/weights.pt"
    model = get_yolo(MODEL_PATH)   # loaded once, cached forever

    frame_iter = iter(frames)
    first = next(frame_iter)
    W, H = first.shape[1], first.shape[0]

    def process(frame):
        results = model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            detections.append({
                "label": model.names[int(box.cls[0])],
                "conf":  round(float(box.conf[0]), 3),
                "box": {
                    "x": round(x1 / W, 4), "y": round(y1 / H, 4),
                    "w": round((x2 - x1) / W, 4), "h": round((y2 - y1) / H, 4),
                },
            })
        emit_detections(dest_cam, detections, W, H)

    process(first)
    for frame in frame_iter:
        process(frame)
```

Then add to `demo1.py`:
```python
from models.my_model.my_model import my_model

MODEL_CONFIG = {
    ...
    "my_model": {"enabled": True, "fn": my_model},
}
```

---

## Detection Event Format

Sent via Socket.IO to Node.js `/camera` namespace, then broadcast to Angular:

```json
{
  "streamId": "cam2sub",
  "ts": 1712345678000,
  "width": 1280,
  "height": 720,
  "detections": [
    {
      "id": "42",
      "label": "fire",
      "conf": 0.91,
      "box": { "x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4 }
    }
  ]
}
```

`streamId` must match the `streamId` input of the Angular `<app-video-viewer>` component.  
All box values are normalised `[0..1]`.

---

## Author

**OranextAI — Smart Factory 4.0 Team**  
https://github.com/OranextAI

## License

© 2025 OranextAI. All rights reserved.
