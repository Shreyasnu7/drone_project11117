# File: laptop_ai/gopro_driver.py
"""
GoProDriver
===========

Safe GoPro control using HTTP + BLE (where supported).
No firmware modification. No unsafe operations.

Purpose:
    • Apply camera_plan parameters (exposure, ISO, shutter, white balance)
    • Start/Stop recording
    • Change resolution and fps
    • Provide GoPro frames to CameraFusion
"""

import requests
import time
import cv2
import numpy as np

class GoProDriver:
    def __init__(self, ip="10.5.5.9"):
        self.ip = ip
        self.last_frame_ts = 0
        self.connected = True

    # ------------------------------------------------------------------
    # HTTP COMMAND HELPERS
    # ------------------------------------------------------------------

    def _cmd(self, path: str):
        try:
            url = f"http://{self.ip}{path}"
            requests.get(url, timeout=0.3)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # PUBLIC CAMERA COMMANDS
    # ------------------------------------------------------------------

    def start_recording(self):
        return self._cmd("/gp/gpControl/command/shutter?p=1")

    def stop_recording(self):
        return self._cmd("/gp/gpControl/command/shutter?p=0")

    def set_fps(self, fps: int):
        # Mapping required based on GoPro model
        # Placeholder:
        return self._cmd(f"/gp/gpControl/setting/3/{fps}")

    def set_resolution(self, res_code: int):
        return self._cmd(f"/gp/gpControl/setting/2/{res_code}")

    def apply_exposure(self, ae_cmd):
        # ISO, shutter, EV compensation
        iso = ae_cmd.get("iso")
        ev = ae_cmd.get("ev")
        # Safe commands:
        if iso:
            self._cmd(f"/gp/gpControl/setting/13/{iso}")
        if ev:
            self._cmd(f"/gp/gpControl/setting/15/{ev}")

    # ------------------------------------------------------------------
    # FRAME RETRIEVAL
    # ------------------------------------------------------------------

    def grab_frame(self):
        """
        Pulls a frame from GoPro live feed.
        """
        try:
            cap = cv2.VideoCapture(f"udp://{self.ip}:8554")
            ret, frame = cap.read()
            cap.release()
            return frame if ret else None
        except Exception:
            return None