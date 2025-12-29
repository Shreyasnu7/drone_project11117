# laptop_ai/camera_brain/ai_color.py
import numpy as np
import cv2

class AIColorController:
    """
    AI color correction, WB, and cinematic LUT suggestion.
    """

    def compute_color_profile(self, frame):
        wb = self._white_balance(frame)
        lut = self._choose_lut(frame)
        return {"white_balance": wb, "lut": lut}

    def _white_balance(self, frame):
        avg = frame.mean(axis=(0,1))
        gain = avg.mean() / avg
        return gain.tolist()

    def _choose_lut(self, frame):
        sat = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[...,1])
        if sat < 50:
            return "film_soft"
        if sat < 100:
            return "teal_orange"
        return "action_punch"
