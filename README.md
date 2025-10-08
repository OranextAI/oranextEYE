
# 📦 OranextEYE - Custom Python Library

**Developed by OranextAI**

A Python library for **camera streaming, AI processing (fire detection), and Kafka alerts**.  
This library provides reusable modules for:

- Pulling RTSP/video streams (`puller.py`)  
- Pushing processed frames to RTSP (`pusher.py`)  
- Sending notifications to Kafka (`notifier.py`)  
- Integrating AI models for detection (`models/`)  

---

## 🗂️ Project Structure

```
oranextEYE/
│
├── oureyes/                 # Custom library
│   ├── __init__.py
│   ├── puller.py            # pull_stream(cam_name)
│   ├── pusher.py            # push_stream(frames, width, height, fps, cam_name)
│   ├── notifier.py          # notify_server(topic, data)
│   └── utils.py             # build RTSP URLs etc.
│
├── models/                  # AI models
│   └── fire_detection/      # Fire detection model module
│       └── aziz.py          # Fire detection model
│       └── test.py          # Usage example
│
├── main.py                  # Optional main script
├── setup.py                 # pip installable package
├── .env.example             # Environment variables example
└── .env                     # Environment variables (copy from .env.example)
```

---

## ⚡ Features

1. **Pull RTSP/video streams** with automatic reconnect:  

```python
from oureyes.puller import pull_stream
frames = pull_stream("fakecamera")
```

2. **Push processed frames to RTSP destination**:  

```python
from oureyes.pusher import push_stream
push_stream(frames, width, height, fps, "pulledcam")
```

3. **Send Kafka notifications** (singleton producer for efficiency):  

```python
from oureyes.notifier import notify_server
notify_server("camera", {"data": "fire_detected", "iddevice": 19, "idattribute": 6})
```

4. **Environment-configurable** via `.env` (Kafka server, RTSP host/port)

---

## 📥 Download / Clone Repository

You can **clone the repository** directly from GitHub:

```bash
git clone https://github.com/OranextAI/oranextEYE.git
cd oranextEYE
```

Or **download as a ZIP** from:  
[https://github.com/OranextAI/oranextEYE](https://github.com/OranextAI/oranextEYE)

---

## 🔧 Setup Environment

1. Create `.env` file from example:

```bash
cp .env.example .env
```

2. Create Python environment (conda recommended):

```bash
conda create -n nefzi_world python=3.10
conda activate nefzi_world
```

3. Install the library in editable mode:

```bash
pip install -e .
```

> This allows you to import `oureyes` from anywhere and update the code live.

---

## 📝 Notes

- **FPS and camera names** can be set in scripts or passed dynamically to functions.  
- `notify_server()` uses a **singleton Kafka producer** to avoid reconnecting for each message.  
- AI models can be stored in `models/` and imported as needed.  
- This setup supports **multiple cameras** and **multiple RTSP outputs**.  

---

## 🎬 Usage Example

Usage is provided in **`models/fire_detection/test.py`**:

```python
from oureyes.puller import pull_stream
from oureyes.pusher import push_stream
from oureyes.notifier import notify_server
from models.fire_detection.aziz import detect_fire

SRC_CAM = "fakecamera"
DEST_CAM = "pulledcam"
FPS = 25

frames = pull_stream(SRC_CAM)
first_frame = next(frames)
height, width = first_frame.shape[:2]

def processed_frames():
    yield detect_fire(first_frame)
    for f in frames:
        processed = detect_fire(f)

        # Example: send Kafka alert
        notify_server("camera", {
            "data": "fire_detected",
            "iddevice": 19,
            "idattribute": 6
        })

        yield processed

push_stream(processed_frames(), width, height, FPS, DEST_CAM)
```

---

## 🚀 Running

```bash
conda activate nefzi_world
python models/fire_detection/test.py
```

- Stream from `SRC_CAM`  
- Apply AI detection  
- Push to `DEST_CAM`  
- Send Kafka notifications for events
