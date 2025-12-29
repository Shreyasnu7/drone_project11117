# File: laptop_ai/ai_noise_reduction.py
import cv2
import numpy as np

class AINoiseReduction:

    def __init__(self):
        self.last_frame = None

    def denoise(self, frame):
        if self.last_frame is None:
            self.last_frame = frame
            return frame

        # Temporal average
        blended = cv2.addWeighted(frame, 0.7, self.last_frame, 0.3, 0)

        # Spatial denoise
        denoised = cv2.fastNlMeansDenoisingColored(blended, None, 3, 3, 7, 21)

        self.last_frame = frame
        return denoised