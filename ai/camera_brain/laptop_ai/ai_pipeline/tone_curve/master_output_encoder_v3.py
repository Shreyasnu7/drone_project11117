
import cv2
import numpy as np

class MasterOutputEncoderV3:
    """
    Final Output Transform (ODT) for Delivery.
    Converts ACES Linear/Log to Rec.709 Gamma 2.4.
    """
    def __init__(self, gamma=2.4):
        self.gamma = gamma

    def apply(self, frame):
        # Normalize
        img_f = frame.astype(np.float32) / 255.0
        
        # Apply Gamma
        # Avoid div zero
        img_f = np.power(np.maximum(img_f, 0), 1.0/self.gamma)
        
        # Scale back
        return np.clip(img_f * 255.0, 0, 255).astype(np.uint8)
