# MediaMTX Configuration

MediaMTX is the camera stream server. It handles RTSP ingestion, WebRTC delivery, HLS, and recording.

## Files

- `mediamtx.yml` — main configuration (edit this file)
- `sync-config.sh` — syncs config to `~/mediamtx-config/` and restarts the container

## Docker Setup

MediaMTX runs as a Docker container. The config and recordings are mounted from the host.

### First-time setup

```bash
# 1. Create required directories
mkdir -p ~/mediamtx-config
mkdir -p ~/recordings

# 2. Copy config to home (snap Docker can only access home directory)
cp /var/www/oranextWEB/oranextEYE/media-config/mediamtx.yml ~/mediamtx-config/mediamtx.yml

# 3. Start the container
sudo docker run -d \
  --name mediamtx \
  --restart unless-stopped \
  --network host \
  -v "$HOME/mediamtx-config/mediamtx.yml:/mediamtx.yml:ro" \
  -v "$HOME/recordings:/recordings" \
  -e MTX_RTSPTRANSPORTS=tcp \
  -e MTX_WEBRTCADDITIONALHOSTS=102.110.12.218 \
  bluenviron/mediamtx:latest-ffmpeg
```

`--restart unless-stopped` means the container starts automatically on system reboot.

### After editing mediamtx.yml

```bash
cd /var/www/oranextWEB/oranextEYE/media-config
bash sync-config.sh
```

This copies the updated config to `~/mediamtx-config/` and restarts the container.

### Useful commands

```bash
# Check status
sudo docker ps | grep mediamtx

# View logs
sudo docker logs mediamtx -f

# Restart
sudo docker restart mediamtx

# Stop
sudo docker stop mediamtx
```

## Ports

| Port | Protocol | Purpose |
|------|----------|---------|
| 8554 | TCP | RTSP |
| 8888 | HTTP | HLS |
| 8889 | HTTP | WebRTC (WHEP/WHIP) |
| 8189 | UDP | WebRTC ICE |
| 9996 | HTTP | Playback server |
| 9997 | HTTP | Control API |
| 1935 | TCP | RTMP |

## Recording

Recordings are saved to `~/recordings/<camera_name>/` on the host.

Format: `fmp4` (fragmented MP4), one segment per hour by default.

## Playback

The playback server runs on `:9996`. It is proxied via nginx at:
`https://vivo.oranextai.run.place/playback/`

The web app accesses recordings through `/api/recordings/:camera` which proxies the MediaMTX API.

## Fake Cameras (looped local video)

You can simulate a camera by looping a local video file through ffmpeg.

### Setup

**1. Create the videos directory on the host:**
```bash
mkdir -p ~/videos
```

**2. Download your video from Google Drive to the server:**
```bash
# Install gdown if not already installed
pip3 install gdown

# Use --fuzzy with the standard Drive share link (works for large files too)
# Share link: https://drive.google.com/file/d/FILE_ID/view
gdown --fuzzy "https://drive.google.com/file/d/FILE_ID/view" -O ~/videos/your-video.mp4
```

**3. Stop the current container and re-run it with the videos volume mounted:**
```bash
sudo docker stop mediamtx && sudo docker rm mediamtx

sudo docker run -d \
  --name mediamtx \
  --restart unless-stopped \
  --network host \
  -v "$HOME/mediamtx-config/mediamtx.yml:/mediamtx.yml:ro" \
  -v "$HOME/recordings:/recordings" \
  -v "$HOME/videos:/videos" \
  -e MTX_RTSPTRANSPORTS=tcp \
  -e MTX_WEBRTCADDITIONALHOSTS=102.110.12.218 \
  bluenviron/mediamtx:latest-ffmpeg
```

### Adding a fake camera

In `mediamtx.yml`, add a new entry under `paths:`:

```yaml
  my_fakecam:
    source: publisher
    overridePublisher: yes
    runOnInit: >
      ffmpeg -re -stream_loop -1 -i /videos/your-video.mp4
      -c:v libx264 -preset veryfast -tune zerolatency
      -profile:v baseline -g 30 -pix_fmt yuv420p -an
      -f rtsp rtsp://127.0.0.1:8554/my_fakecam
    runOnInitRestart: yes
    record: yes
    recordPath: /recordings/%path/%Y-%m-%d_%H-%M-%S-%f
```

- Change `my_fakecam` to whatever name you want (this becomes the stream path)
- Change `your-video.mp4` to the actual filename in `~/videos/`
- `runOnInitRestart: yes` ensures the stream auto-restarts if ffmpeg crashes
- The stream will be available at `rtsp://SERVER_IP:8554/my_fakecam` and all other protocols (HLS, WebRTC) as usual

Then apply:
```bash
bash sync-config.sh
```

---

## Real Cameras (RTSP)

### Adding a real camera

In `mediamtx.yml`, add a new entry under `paths:`:

```yaml
  cam3:
    source: rtsp://admin:password@192.168.1.102:554/Streaming/Channels/101
    sourceProtocol: tcp
    sourceOnDemand: no
    runOnInitRestart: no
    record: yes
    recordPath: /recordings/%path/%Y-%m-%d_%H-%M-%S-%f
```

- Change `cam3` to the camera name (used as the stream path)
- Replace the RTSP URL with your camera's actual URL — check your camera's manual or admin panel for the correct stream URL
- `admin:password` — replace with your camera's username and password
- `192.168.1.102` — replace with your camera's IP address
- Common RTSP URL patterns by brand:
  - Hikvision main stream: `rtsp://user:pass@IP:554/Streaming/Channels/101`
  - Hikvision sub stream: `rtsp://user:pass@IP:554/Streaming/Channels/102`
  - Dahua: `rtsp://user:pass@IP:554/cam/realmonitor?channel=1&subtype=0`
  - Generic: `rtsp://user:pass@IP:554/live/ch0`

Then apply:
```bash
bash sync-config.sh
```

---

## VPS IP

The WebRTC additional host is set to `102.110.12.218` (VPS public IP).
If the IP changes, update `MTX_WEBRTCADDITIONALHOSTS` in the docker run command and re-run it.
