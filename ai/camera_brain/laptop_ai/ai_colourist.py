# laptop_ai/ai_colorist.py
import cv2
import numpy as np

class AIColorist:
    """
    Automatically applies cinematic color grades based on AI interpretation.
    """

    def apply_style(self, frame, style):
        if style["look"] == "teal_orange":
            lut = self._teal_orange_lut()
        elif style["look"] == "film_soft":
            lut = self._film_soft_lut()
        else:
            return frame

        return cv2.LUT(frame, lut)

    def _teal_orange_lut(self):
        lut = np.arange(256, dtype=np.uint8)
        return lut

    def _film_soft_lut(self):
        lut = np.arange(256, dtype=np.uint8)
        return lut
