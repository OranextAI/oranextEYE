ffmpeg -re -stream_loop -1 -i video.mp4 \
       -c:v libx264 -f rtsp \
       rtsp://localhost:8554/fakecamera
