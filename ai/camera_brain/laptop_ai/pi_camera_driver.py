# File: laptop_ai/pi_camera_driver.py
"""
PiCameraDriver (Internal Camera)
================================

Provides:
    • Safe control of internal camera
    • Exposure + focus + white balance
    • HDR (optional)
    • Sends frames/depth/flow to CameraFusion
"""

import cv2
import numpy as np
import time

class PiCameraDriver:

    def __init__(self, device=0, fusion=None):
        self.device = device
        self.cap = cv2.VideoCapture(device)
        self.fusion = fusion
        self.last_frame_ts = 0

    def grab(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        # Optionally depth + optical flow future modules insert here:
        depth = None
        flow = None

        if self.fusion:
            self.fusion.update_internal_frame(frame, depth, flow)

        return frame

    def apply_autofocus(self, af_cmd):
        # For manual-focus modules like IMX678, adjust lens actuator here
        pass

    def apply_exposure(self, ae_cmd):
        # Set brightness / shutter / ISO (if supported)
        if "brightness" in ae_cmd:
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, ae_cmd["brightness"])

    def release(self):
        self.cap.release()