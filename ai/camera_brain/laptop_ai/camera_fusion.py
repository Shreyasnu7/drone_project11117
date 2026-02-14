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

# Wired Fusion Components
try:
    from laptop_ai.ai_fusion_pipeline import AIFusionPipeline
    from laptop_ai.pi_camera_driver import PiCameraDriver
    from laptop_ai.threaded_camera import CameraStream
except ImportError:
    AIFusionPipeline = None
    PiCameraDriver = None
    CameraStream = None


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

        # Fusion Engine
        self.fusion_pipeline = AIFusionPipeline() if AIFusionPipeline else None

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
        
    # Alias for Director compatibility
    update_external_frame = update_gopro_frame

    def get_active_frame(self) -> Optional[np.ndarray]:
        """Returns the frame from the currently selected best camera."""
        source = self.select_best_source()
        if source == "gopro":
            return self.last_gopro_frame
        return self.last_internal_frame

    def update_gyro(self, gx, gy, gz):
        self.last_gyro = {"gx": gx, "gy": gy, "gz": gz}
        self.last_update_ts = time.time()

    # ----------------------------------------------------------------------
    # FUSION LOGIC
    # ----------------------------------------------------------------------

    def select_best_source(self) -> str:
        """
        Decides which camera is 'Better' right now.
        Returns: 'internal' or 'gopro'
        """
        score_internal = 0
        score_gopro = 0
        
        # 1. Availability check
        if self.last_internal_frame is not None: score_internal += 50
        if self.last_gopro_frame is not None: score_gopro += 50
        
        # 2. Recency check (penalize stale frames > 0.5s)
        now = time.time()
        # Would need per-frame timestamps, ignoring for simple fusing
        
        # 3. Quality / Capability Bias
        # GoPro generally has better optics
        if self.last_gopro_frame is not None: score_gopro += 20
        
        # Internal has Depth/Flow metadata usually
        if self.last_depth_map is not None: score_internal += 30
        
        if score_gopro > score_internal: return "gopro"
        return "internal"

    def get_fusion_state(self) -> Dict[str, Any]:
        """
        Returns a standardized fusion state dictionary.
        This is consumed by AICameraBrain and camera_selector.
        """
        best_cam = self.select_best_source()

        return {
            "timestamp": time.time(),
            "gyro": self.last_gyro,
            "frames": {
                "internal": self.last_internal_frame,
                "gopro": self.last_gopro_frame
            },
            "selected_camera": best_cam, # The Brain uses this choice
            "depth": self.last_depth_map,
            "flow": self.last_flow,
            "meta": {
                "internal_rs_ms": self.internal_rs_offset,
                "gopro_rs_ms": self.gopro_rs_offset
            }
        }

# Alias for compatibility with modules expecting 'FusionEngine'
FusionEngine = CameraFusion