# laptop_ai/camera_brain/ai_sharpness.py
import cv2
import numpy as np

class AISharpnessEngine:

    def adjust(self, frame):
        return cv2.GaussianBlur(frame, (0,0), 1.2)
