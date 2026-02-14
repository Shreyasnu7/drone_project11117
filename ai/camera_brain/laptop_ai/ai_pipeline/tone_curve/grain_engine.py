
import cv2
import numpy as np

class GrainEngine:
    """
    Organic 35mm Film Grain Generator.
    """
    def __init__(self, intensity=0.05):
        self.intensity = intensity

    def apply(self, frame):
        h, w, c = frame.shape
        # Gaussian noise
        noise = np.random.normal(0, self.intensity * 255, (h, w, c)).astype(np.float32)
        
        # Add to frame
        noisy_frame = frame.astype(np.float32) + noise
        
        # Clip
        return np.clip(noisy_frame, 0, 255).astype(np.uint8)
