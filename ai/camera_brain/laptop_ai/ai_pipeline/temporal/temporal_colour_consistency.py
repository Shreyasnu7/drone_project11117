
import cv2
import numpy as np

class TemporalConsistency:
    """
    Prevents color flickering by temporally smoothing the global frame statistics.
    """
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.accum_avg = None # Moving average of frame bgr
        
    def process(self, frame: np.ndarray) -> np.ndarray:
        if frame is None: return None
        
        # 1. Current Frame Global Mean
        curr_mean = np.array(cv2.mean(frame)[:3])
        
        if self.accum_avg is None:
            self.accum_avg = curr_mean
            return frame
            
        # 2. Update Moving Average
        self.accum_avg = self.accum_avg * (1 - self.alpha) + curr_mean * self.alpha
        
        # 3. Gain Correction (Match Scale)
        # target / current
        gain = self.accum_avg / (curr_mean + 1e-6)
        
        # 4. Apply Gain (Optimization: multiply global scalar)
        # This is expensive per-pixel in python, better used for exposure feedback.
        # For software correction:
        frame_float = frame.astype(np.float32)
        frame_float *= gain # Broadcast multiply
        
        return np.clip(frame_float, 0, 255).astype(np.uint8)
