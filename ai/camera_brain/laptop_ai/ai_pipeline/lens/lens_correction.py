
import cv2
import numpy as np

class LensSignalEngine:
    """
    Corrects lens distortion for GoPro / Wide Angle lenses.
    Uses coefficient method.
    """
    def __init__(self, k1=-0.2, k2=0.0):
        self.k1 = k1
        self.k2 = k2
        self.map_x = None
        self.map_y = None
        
    def compute_compression(self, distance, compression_bias):
        """
        Mock physics calc for lens compression feel.
        """
        return 1.0 + (10.0 / (distance + 0.1)) * compression_bias
        
    def undistort(self, frame: np.ndarray) -> np.ndarray:
        if frame is None: return None
        h, w = frame.shape[:2]
        
        # Lazy init maps
        if self.map_x is None or self.map_x.shape[:2] != (h, w):
            # Camera Matrix (Approx center)
            K = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]])
            D = np.array([self.k1, self.k2, 0, 0])
            
            # New K (retain scale)
            K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w,h), np.eye(3), balance=1)
            self.map_x, self.map_y = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K_new, (w,h), cv2.CV_16SC2)
            
        return cv2.remap(frame, self.map_x, self.map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
