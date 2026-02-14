
import cv2
import numpy as np

class DynamicRangeOptimizer:
    """
    Enhances local contrast while protecting highlights.
    Uses CLAHE (Contrast Limited Adaptive Histogram Equalization).
    """
    def __init__(self, clip_limit=2.0, tile_size=(8,8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        
    def process(self, frame: np.ndarray) -> np.ndarray:
        if frame is None: return None
        
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L-channel
        l_clahe = self.clahe.apply(l)
        
        # Merge back
        lab_new = cv2.merge((l_clahe, a, b))
        
        # Convert back to BGR
        return cv2.cvtColor(lab_new, cv2.COLOR_LAB2BGR)
