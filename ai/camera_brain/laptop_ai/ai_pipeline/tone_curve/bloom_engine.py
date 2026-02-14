
import cv2
import numpy as np

class BloomEngine:
    """
    Cinematic Bloom Engine.
    Creates a soft glow for bright highlights.
    """
    def __init__(self, threshold=0.8, radius=15, strength=0.5):
        self.threshold = threshold * 255
        self.radius = radius
        self.strength = strength

    def apply(self, frame):
        # 1. Thresholding to extract highlights
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)[1]
        
        # 2. Blur the highlights
        blur = cv2.GaussianBlur(frame, (0, 0), self.radius)
        mask_f = (mask.astype(float) / 255.0)[:,:,None]
        
        # 3. Additive blending
        # Only add blur where mask is active
        glow = blur * mask_f * self.strength
        
        result = cv2.add(frame, glow.astype(np.uint8))
        return result
