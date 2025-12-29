# File: laptop_ai/camera_fusion.py
"""
CameraFusion
============

This module fuses data coming from:
    • Internal AI camera (PiCam or IMX678)
    • GoPro (via GoProDriver)
    • Gyro/IMU from Radxa
    • AI metadata (depth, optical flow, stabilization transforms)

It produces a unified `fusion_state` dictionary used by:
    • AICameraBrain
    • camera_selector
    • Stabilizer
    • Director pipeline
"""

import time
import numpy as np
from typing import Dict, Any, Optional


class CameraFusion:

    def __init__(self):
        self.last_internal_frame = None
        self.last_gopro_frame = None
        self.last_depth_map = None
        self.last_flow = None
        self.last_gyro = {"gx": 0.0, "gy": 0.0, "gz": 0.0}

        self.last_update_ts = 0

        # Rolling shutter offsets (ms) used for advanced stabilization
        self.internal_rs_offset = 0.0
        self.gopro_rs_offset = 0.0

    # ----------------------------------------------------------------------
    # UPDATE METHODS (called by camera drivers)
    # ----------------------------------------------------------------------

    def update_internal_frame(self, frame: np.ndarray, depth=None, flow=None):
        self.last_internal_frame = frame
        if depth is not None:
            self.last_depth_map = depth
        if flow is not None:
            self.last_flow = flow
        self.last_update_ts = time.time()

    def update_gopro_frame(self, frame: np.ndarray):
        self.last_gopro_frame = frame
        self.last_update_ts = time.time()

    def update_gyro(self, gx, gy, gz):
        self.last_gyro = {"gx": gx, "gy": gy, "gz": gz}
        self.last_update_ts = time.time()

    # ----------------------------------------------------------------------
    # FUSION LOGIC
    # ----------------------------------------------------------------------

    def get_fusion_state(self) -> Dict[str, Any]:
        """
        Returns a standardized fusion state dictionary.
        This is consumed by AICameraBrain and camera_selector.
        """

        return {
            "timestamp": time.time(),
            "gyro": self.last_gyro,
            "frames": {
                "internal": self.last_internal_frame,
                "gopro": self.last_gopro_frame
            },
            "depth": self.last_depth_map,
            "flow": self.last_flow,
            "meta": {
                "internal_rs_ms": self.internal_rs_offset,
                "gopro_rs_ms": self.gopro_rs_offset
            }
        }