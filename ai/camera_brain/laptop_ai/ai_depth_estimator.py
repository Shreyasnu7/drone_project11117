# File: laptop_ai/ai_depth_estimator.py
import numpy as np

class AIDepthEstimator:

    def __init__(self):
        # placeholder for depth model
        pass

    def estimate(self, frame):
        h, w = frame.shape[:2]

        # Placeholder: fake smooth gradient depth
        depth = np.tile(np.linspace(1, 0, w), (h,1))

        # Center subject mask (placeholder)
        mask = np.zeros((h,w), dtype=np.uint8)
        cx, cy = w//2, h//2
        mask[cy-80:cy+80, cx-80:cx+80] = 255

        return depth, mask