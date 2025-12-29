# laptop_ai/ai_lensfix.py
import cv2
import numpy as np

class LensCorrector:
    """
    Removes GoPro fisheye distortion using calibrated intrinsics.
    """

    def __init__(self):
        self.K = np.array([[720, 0, 640],
                           [0, 720, 360],
                           [0,   0,   1]])
        self.D = np.array([-0.32, 0.12, -0.001, 0.0])

    def correct(self, frame):
        h, w = frame.shape[:2]
        newK, _ = cv2.getOptimalNewCameraMatrix(self.K, self.D, (w, h), 0)
        return cv2.undistort(frame, self.K, self.D, None, newK)
