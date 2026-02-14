
import cv2
import numpy as np

class HalationEngine:
    """
    Simulates film halation (red-orange glow around highlights).
    Caused by light reflecting off the film base back into the red emulsion layer.
    """
    def __init__(self, threshold=0.9, radius=10, strength=0.4):
        self.threshold = threshold * 255
        self.radius = radius
        self.strength = strength

    def apply(self, frame):
        # 1. Isolate Red Channel Highlights
        b, g, r = cv2.split(frame)
        mask = cv2.threshold(r, self.threshold, 255, cv2.THRESH_BINARY)[1]
        
        # 2. Blur the Red Channel Mask
        halation = cv2.GaussianBlur(mask, (0, 0), self.radius)
        halation = halation.astype(np.float32) / 255.0
        
        # 3. Add back to Red Channel (and slightly to Green for orange tint)
        r_new = r.astype(np.float32) + (halation * 255.0 * self.strength)
        g_new = g.astype(np.float32) + (halation * 255.0 * self.strength * 0.3)
        
        frame_out = cv2.merge([b, np.clip(g_new, 0, 255).astype(np.uint8), np.clip(r_new, 0, 255).astype(np.uint8)])
        return frame_out
