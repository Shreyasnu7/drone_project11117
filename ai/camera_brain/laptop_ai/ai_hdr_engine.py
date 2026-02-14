# File: laptop_ai/ai_hdr_engine.py
"""
AI HDR Engine
-------------
Merges multiple exposures into a single HDR frame.
Used by internal and GoPro cameras.
"""

import cv2
import numpy as np

class AIHDREngine:

    def __init__(self):
        # OpenCV HDR components
        self.merge_debevec = cv2.createMergeDebevec()
        self.tonemap = cv2.createTonemapDrago(1.0, 0.7)

    def compute_hdr(self, frames, exposures):
        """
        frames: [frame1, frame2, frame3]
        exposures: corresponding exposure times
        """

        if len(frames) < 2:
            return frames[0]

        try:
            hdr = self.merge_debevec.process(frames, exposures)
            ldr = self.tonemap.process(hdr)
            ldr_8bit = np.clip(ldr * 255, 0, 255).astype(np.uint8)
            return ldr_8bit
        except Exception:
            return frames[0]