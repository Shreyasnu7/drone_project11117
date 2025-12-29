# laptop_ai/camera_brain/ai_exposure.py
import numpy as np
import cv2

class AIExposureController:
    """
    AI exposure/ISO/shutter controller for both GoPro and internal camera.
    """

    def compute_exposure(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)

        # Target brightness range for cinematic look
        target = 0.45

        # Exposure compensation (-2 to +2)
        ev = (target - (brightness / 255.0)) * 4
        ev = np.clip(ev, -2.0, 2.0)

        return {
            "ev": float(ev),
            "iso": self._recommend_iso(brightness),
            "shutter": self._recommend_shutter(brightness)
        }

    def _recommend_iso(self, brightness):
        if brightness < 40:
            return 1600
        if brightness < 70:
            return 800
        if brightness < 120:
            return 400
        return 100

    def _recommend_shutter(self, brightness):
        if brightness < 50:
            return "1/48"
        if brightness < 100:
            return "1/96"
        return "1/192"
