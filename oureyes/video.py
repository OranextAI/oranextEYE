import os
import cv2
from dotenv import load_dotenv

# Load the .env file that sits in the same directory as this script
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    print("⚠️ Warning: .env file not found in oureyes folder")

def getvideo(cam_name: str):
    """
    Open and display an RTSP video stream using OpenCV.
    RTSP URL is built from environment variables HOST and PORT.
    Example: rtsp://HOST:PORT/cam
    """
    host = os.getenv("HOST")
    port = os.getenv("PORT")

    if not host or not port:
        print("❌ Missing HOST or PORT in .env")
        return

    rtsp_url = f"rtsp://{host}:{port}/{cam_name}"
    print(f"🎥 Connecting to {rtsp_url} ...")

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("❌ Could not open the RTSP stream.")
        return

    print("✅ Stream opened successfully. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Stream ended or cannot read frame.")
            break

        cv2.imshow(f"Camera: {cam_name}", frame)

        # quit with q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🔒 Stream closed.")
