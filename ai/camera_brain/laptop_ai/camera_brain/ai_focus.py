# laptop_ai/camera_brain/ai_focus.py
import cv2
import numpy as np

class AIFocusEngine:
    """
    Digital focus estimation + real focus hints for PiCam with autofocus lens.
    """

    def estimate_focus(self, frame):
        laplacian = cv2.Laplacian(frame, cv2.CV_64F).var()
        return laplacian

    def compute_focus_adjustment(self, focus_metric):
        if focus_metric < 60:
            return "increase_focus"
        if focus_metric > 200:
            return "decrease_focus"
        return "hold"
