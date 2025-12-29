# laptop_ai/pi_camera.py
import cv2
import time
import asyncio
from typing import Optional

# Simple RTSP / v4l2 capture wrapper for Pi camera v2.
class PiCamera:
    def __init__(self, device=0, width=1280, height=720, fps=30):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

    def start(self):
        # If using libcamera / v4l2, device may be 0 or /dev/video0
        self.cap = cv2.VideoCapture(self.device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

    def read(self, timeout=1.0) -> Optional[any]:
        if self.cap is None:
            self.start()
        t0 = time.time()
        while time.time() - t0 < timeout:
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None

    def stop(self):
        if self.cap:
            self.cap.release()
            self.cap = None
