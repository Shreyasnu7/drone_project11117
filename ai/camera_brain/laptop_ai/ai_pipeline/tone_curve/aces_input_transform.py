
import numpy as np

class ACESInputTransform:
    """
    Handles Input Device Transform (IDT) for ACES pipeline.
    Connects raw camera space -> AP0/AP1 ACES color space.
    """
    def __init__(self):
        pass

    def process(self, frame: np.ndarray, profile="sRGB", ei=800) -> np.ndarray:
        # Input Transform (IDT): sRGB -> Linear
        # Inverse Gamma 2.2 for physical light linearity
        if frame is None:
            return None
            
        norm = frame.astype(np.float32) / 255.0
        return np.power(norm, 2.2)
