import os


OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Server Config
VPS_WS = os.getenv("VPS_WS", "wss://drone-server-r0qe.onrender.com/ws/connect/laptop_ai")
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "dev_token_123")
API_BASE = os.getenv("API_BASE", "https://drone-server-r0qe.onrender.com")

# YOLO model path (we default to yolov8n.pt). Put the .pt file in laptop_ai/ or change path.
YOLO_MODEL_PATH = "yolov8n.pt"

# RTSP / camera input for laptop (set to your Radxa RTSP or local camera index)
RTSP_URL = "rtsp://YOUR_VPS_IP:8554/drone_feed"  # set to actual stream or "0" for local webcam.

# Performance config
AI_CALL_INTERVAL = 4.0   # seconds between full AI cloud calls for same job
FRAME_SKIP = 2           # process every Nth frame for vision (increase to save CPU)

# artifact dir
TEMP_ARTIFACT_DIR = "./artifacts"
MAX_IMAGES_UPLOAD = 25
MAX_VIDEO_UPLOAD_MB = 100
