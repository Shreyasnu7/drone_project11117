# laptop_ai/camera_brain/ai_filters.py
import cv2
import numpy as np

class AIFilterEngine:

    def apply(self, frame, params):
        sat = params.get("saturation", 1.1)
        con = params.get("contrast", 1.05)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[...,1] = np.clip(hsv[...,1] * sat, 0, 255)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        frame = cv2.convertScaleAbs(frame, alpha=con, beta=0)

        return frame
