
import cv2
import numpy as np

class LabFlickerEngineV4:
    """
    Simulates chemical bath exposure flicker (subtle brightness oscillation).
    """
    def __init__(self, intensity=0.02):
        self.intensity = intensity

    def apply(self, frame):
        # Random gain
        gain = 1.0 + np.random.uniform(-self.intensity, self.intensity)
        return cv2.multiply(frame, gain)
