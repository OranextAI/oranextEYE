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

## VPS IP

The WebRTC additional host is set to `102.110.12.218` (VPS public IP).
If the IP changes, update `MTX_WEBRTCADDITIONALHOSTS` in the docker run command and re-run it.
