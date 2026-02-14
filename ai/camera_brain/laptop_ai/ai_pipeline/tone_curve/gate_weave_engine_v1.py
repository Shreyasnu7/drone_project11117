
import cv2
import numpy as np

class GateWeaveEngineV1:
    """
    Simulates mechanical film gate weave (projector instabilty).
    Applies tiny random translations to the frame.
    """
    def __init__(self, amplitude=1.5):
        self.amplitude = amplitude

    def apply(self, frame):
        h, w = frame.shape[:2]
        # Random shift
        dx = np.random.uniform(-self.amplitude, self.amplitude)
        dy = np.random.uniform(-self.amplitude, self.amplitude)
        
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
