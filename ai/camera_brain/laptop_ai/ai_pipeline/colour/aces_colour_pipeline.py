
import cv2
import numpy as np

class ACESColorPipeline:
    """
    Real-time ACES (Academy Color Encoding System) emulation for sRGB cameras.
    """
    def __init__(self):
        # Constants for approximate ACES RRT
        self.a = 2.51
        self.b = 0.03
        self.c = 2.43
        self.d = 0.59
        self.e = 0.14

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Input: sRGB frame (uint8)
        Output: ACES-like filmic graded frame (uint8)
        """
        if frame is None: return None
        
        # 1. Normalize to 0-1
        frame_norm = frame.astype(np.float32) / 255.0
        
        # 2. Apply ACES RRT (Reference Rendering Transform) Approximation
        # x * (a*x + b) / (x * (c*x + d) + e)
        numerator = frame_norm * (self.a * frame_norm + self.b)
        denominator = frame_norm * (self.c * frame_norm + self.d) + self.e
        aces = numerator / denominator
        
        # 3. Saturate (ACES tends to desaturate)
        # Convert to HSV, boost Saturation 1.2x
        # But for speed in NumPy:
        # Simple weighted mix with gray world? No, keep it simple.
        
        # 4. Clip & Denormalize
        return np.clip(aces * 255, 0, 255).astype(np.uint8)
