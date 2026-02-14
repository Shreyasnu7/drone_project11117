
import cv2
import numpy as np

class FilmStockSpectralV3:
    """
    Spectral sensitivity emulation for Kodak Vision3 500T.
    Uses channel crosstalk matrix to simulate film emulsion density.
    """
    def __init__(self, stock="500T"):
        self.stock = stock
        # 3x3 Color Matrix for Film Look (Teal shadows, Orange highlights)
        # B_out = a*B + b*G + c*R
        self.matrix = np.array([
            [0.85, 0.10, 0.05], # Blue channel (less sensitive)
            [0.05, 0.85, 0.10], # Green
            [0.02, 0.10, 0.88]  # Red
        ])

    def apply(self, frame):
        # Linear transform
        # OpenCV uses BGR, matrix is BGR
        return cv2.transform(frame, self.matrix)
